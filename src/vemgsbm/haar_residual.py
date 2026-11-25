"""Native LR conditioning and lossless 16-band XY residuals for paired MRI."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import _TimeResidualBlock, batch_time
from .momentum.model import HaarXY
from .paths import linear_path, optimize_path, sample_path


HAAR_MODES = {"haar_direct", "haar_bridge"}


class LRConditionEncoder(nn.Module):
    """A frozen copy of this encoder can describe LR sites without HR input."""

    def __init__(self, widths, multiscale_xy=False):
        super().__init__()
        a, b, c = widths
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1), nn.SiLU(),
                          nn.Conv3d(cout, cout, 3, padding=1))
            for cin, cout, stride in ((1, a, 1), (a, b, (1, 2, 2)), (b, c, (1, 2, 2)))
        ])
        self.multiscale = XYMultiScaleResidual(c) if multiscale_xy else nn.Identity()

    def forward(self, lr, *, descriptor_only=False):
        if lr.ndim != 5 or lr.shape[1] != 1:
            raise ValueError("Expected native LR of shape (B,1,Z,Y,X)")
        if descriptor_only:
            return self.layers[0](lr)
        features = []
        for layer in self.layers:
            lr = layer(lr)
            features.append(lr)
        features[-1] = self.multiscale(features[-1])
        return features


class XYMultiScaleResidual(nn.Module):
    """Expand anatomical context without changing or dilating the Z axis."""
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv3d(channels, channels, (1,3,3), padding=(0,d,d),
                                    dilation=(1,d,d), groups=channels), nn.SiLU(),
                          nn.Conv3d(channels, channels, 1)) for d in (1,2,4)])

    def forward(self, value):
        for block in self.blocks:
            value = value + block(value)
        return value


class HaarResidualNetwork(nn.Module):
    def __init__(self, mode, widths=(48, 96, 128), readout="integral", spatial=None):
        super().__init__()
        if mode not in HAAR_MODES or readout not in {"integral", "endpoint"}:
            raise ValueError("Unknown Haar MRI model or readout")
        self.mode, self.readout = mode, readout
        self.spatial = spatial
        self.codec = HaarXY()
        self.lr_encoder = LRConditionEncoder(widths)
        a, b, c = widths
        embedding = 4 * a
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 2 * a)))
        self.time = nn.Sequential(nn.Linear(embedding, embedding), nn.SiLU(), nn.Linear(embedding, embedding))
        self.direction = nn.Embedding(2, embedding)
        self.state = nn.Conv3d(16, a, 3, padding=1)
        self.first = _TimeResidualBlock(a, a, embedding)
        self.down1 = nn.Conv3d(a, b, 3, stride=(1, 2, 2), padding=1)
        self.second = _TimeResidualBlock(b, b, embedding)
        self.down2 = nn.Conv3d(b, c, 3, stride=(1, 2, 2), padding=1)
        self.deep = _TimeResidualBlock(c, c, embedding)
        self.up2 = _TimeResidualBlock(c + b, b, embedding)
        self.up1 = _TimeResidualBlock(b + a, a, embedding)
        self.endpoint = nn.Conv3d(a, 16, 3, padding=1)
        self.drift = nn.Conv3d(a, 16, 3, padding=1) if mode == "haar_bridge" else None
        if spatial is not None:
            # Learn corrections to both unconstrained heads in residual space.
            # Start from the ungated identity map, including in low-support regions.
            self.geometry_embedding = nn.Sequential(nn.Conv3d(65, a, 1), nn.SiLU())
            self.geometry_delta = nn.Conv3d(a, 32, 1)
            self.geometry_gate = nn.Conv3d(2 * a, 32, 1)
            nn.init.zeros_(self.geometry_delta.weight)
            nn.init.zeros_(self.geometry_delta.bias)
        for head in (self.endpoint, self.drift):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, state, condition, time, direction=0, bound=None):
        angles = batch_time(time, state)[:, None] * self.frequencies.to(state)
        embedding = self.time(torch.cat((angles.sin(), angles.cos()), dim=1))
        if self.direction is not None:
            embedding = embedding + self.direction.weight[direction]
        low, middle, deep = condition
        a = self.first(self.state(state) + low, embedding)
        b = self.second(self.down1(a) + middle, embedding)
        c = self.deep(self.down2(b) + deep, embedding)
        def align(x, skip):
            return torch.cat((F.interpolate(x, size=skip.shape[-3:], mode="trilinear",
                                            align_corners=False), skip), dim=1)
        hidden = self.up1(align(self.up2(align(c, b), embedding), a), embedding)
        clean = self.endpoint(hidden)
        drift = self.drift(hidden) if self.drift is not None else None
        if self.spatial is not None:
            if bound is None:
                raise ValueError("Spatial Haar bridge requires an LR-conditioned chart")
            query = state if self.spatial.get("query", "endpoint") == "state" else clean
            spatial = self.geometry_embedding(bound.features(bound.unwhiten(query)))
            gate = self.geometry_gate(torch.cat((hidden, spatial), dim=1)).sigmoid()
            dc, dd = (gate * self.geometry_delta(spatial)).chunk(2, dim=1)
            clean = clean + bound.whiten(dc)
            drift = drift + bound.whiten(dd)
        return clean, drift

    def zero_state(self, lr, base):
        if lr is None or lr.shape != (*base.shape[:3], base.shape[-2] // 4, base.shape[-1] // 4):
            raise ValueError("Real native LR must align with the XY x4 base and Z grid")
        return lr.new_zeros((len(lr), 16, *lr.shape[2:]))

    def _before_last_step(self, state, condition, steps, sigma, bound=None, generator=None, noise_steps=None):
        for i in range(steps - 1):
            time = max(.02, min(.98, i / steps))
            _, drift = self(state, condition, time, bound=bound)
            state = state + drift / steps
            if sigma:
                noise = (torch.randn(state.shape, dtype=state.dtype, device=state.device, generator=generator)
                         if noise_steps is None else noise_steps[i])
                state = state + noise * (sigma / math.sqrt(steps))
        return state

    @torch.no_grad()
    def restore(self, lr, base, steps, sigma, generator=None, atlas=None, noise_steps=None,
                sampler_type=None, solver="euler", initial_state=None):
        if sampler_type is not None:
            if sampler_type not in {"ode", "sde"} or solver != "euler":
                raise ValueError("Legacy drift checkpoints support Euler ode/sde only")
            if sampler_type == "ode":
                sigma = 0.0
        if not isinstance(steps, int) or steps < 1 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("Invalid Haar bridge sampler steps or sigma")
        state = self.zero_state(lr, base)
        condition = self.lr_encoder(lr)
        bound = atlas.bind(lr) if atlas is not None else None
        if self.mode == "haar_direct":
            residual, _ = self(state, condition, 0.)
        else:
            state = self._before_last_step(state, condition, steps, sigma, bound, generator, noise_steps)
            time = max(.02, min(.98, (steps - 1) / steps))
            endpoint, drift = self(state, condition, time, bound=bound)
            residual = endpoint if self.readout == "endpoint" else state + drift / steps
        return base + self.codec.decode(bound.unwhiten(residual) if bound is not None else residual)


class HaarObjective(nn.Module):
    def __init__(self, model, config, atlas=None, teacher=None):
        super().__init__()
        self.model = model
        self.config = config
        self.atlas = atlas
        self.teacher = teacher

    def forward(self, lr, base, hr, case_ids=None):
        model = self.model
        if self.config.get("train", {}).get("objective") == "sr_reconstruction":
            if getattr(model, "predict_target", None) != "x1":
                raise ValueError("SR reconstruction requires the x1 endpoint model")
            return self._sr_reconstruction_loss(lr, base, hr, case_ids)
        if getattr(model, "predict_target", None) == "x1":
            return self._x1_loss(lr, base, hr, case_ids)
        condition = model.lr_encoder(lr)
        zero = model.zero_state(lr, base)
        if model.mode == "haar_direct":
            clean, _ = model(zero, condition, 0.)
            mse = F.mse_loss(base + model.codec.decode(clean), hr)
            return {"loss": mse, "endpoint_pixel_mse": mse}
        target = model.codec.encode(hr - base)
        bound, path_metric = None, None
        if self.atlas is not None:
            if case_ids is None:
                raise ValueError("TRAIN spatial queries require case IDs for held-out fold routing")
            bound = self.atlas.bind(lr, case_ids)
            target = bound.whiten(target)
            if model.spatial.get("path_geometry", False):
                path_metric = bound.path_metric()
        path_cfg = self.config["path"]
        if path_metric is None:
            # Exact minimum of the mean and variance control action.
            path = linear_path(zero, target, torch.zeros_like(zero), path_cfg["knots"], path_cfg["sigma"])
            diagnostics = {"mean_correction_rms": 0., "accepted_steps": 0}
        else:
            # CondSOC may draw quadrature noise. Preserve the matching-time and
            # noise RNG stream shared with the network-only path ablation.
            devices = [target.device] if target.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                path, diagnostics = optimize_path(zero, target, torch.zeros_like(zero), path_metric, path_cfg)
        cutoff = self.config["train"].get("matching_time_min", .02)
        t = torch.rand(len(hr), device=hr.device) * (1 - 2 * cutoff) + cutoff
        noise = torch.randn_like(target)
        state, f = sample_path(path, t, noise, "forward")
        _, r = sample_path(path, t, noise, "backward")
        clean, forward = model(state, condition, t, 0, bound)
        _, reverse = model(state, condition, t, 1, bound)
        fm, rm = F.mse_loss(forward, f), F.mse_loss(reverse, r)
        mse = F.mse_loss(base + model.codec.decode(bound.unwhiten(clean) if bound is not None else clean), hr)
        weight = self.config["train"].get("endpoint_weight", 1.)
        terms = {"loss": (fm + rm) * .5 + weight * mse, "endpoint_pixel_mse": mse,
                 "forward_matching": fm, "backward_matching": rm,
                 "path_mean_correction_rms": hr.new_tensor(diagnostics["mean_correction_rms"])}
        if self.config["train"].get("matching_units") == "endpoint":
            if bound is not None or path_metric is not None:
                raise ValueError("Endpoint-equivalent matching requires the unwhitened straight Brownian bridge")
            # x + (1-t) f = target and x + t r = source. HaarXY is orthonormal.
            tau = t.view(-1, 1, 1, 1, 1)
            forward_mse = ((1 - tau) * (forward - f)).square().mean()
            reverse_mse = (tau * (reverse - r)).square().mean()
            matching = (forward_mse + reverse_mse) * .5
            terms.update(forward_endpoint_equivalent_mse=forward_mse,
                         reverse_endpoint_equivalent_mse=reverse_mse)
            # Train the supervised readout also on states available at deployment.
            # Sample a subset to bound cost, stop gradients through the simulated trajectory.
            n = max(1, len(lr) // 8)
            cond = [level[:n] for level in condition]
            sampler = self.config["sampling"]
            steps = sampler["steps"]
            with torch.no_grad():
                final_state = model._before_last_step(zero[:n], cond, steps, sampler["inference_sigma"])
            with torch.set_grad_enabled(torch.is_grad_enabled() and weight > 0):
                last_t = max(.02, min(.98, (steps - 1) / steps))
                last_clean, last_drift = model(final_state, cond, last_t)
                coefficients = last_clean if model.readout == "endpoint" else final_state + last_drift / steps
                reconstruction = base[:n] + model.codec.decode(coefficients)
                rollout_mse = F.mse_loss(reconstruction, hr[:n])
            terms["rollout_pixel_mse"] = rollout_mse.detach()
            terms["loss"] = matching + weight * (mse + rollout_mse) * .5
        if bound is not None:
            terms.update({key: torch.as_tensor(value, device=hr.device) for key, value in bound.diagnostics().items()})
            rms = target.square().mean().sqrt().clamp_min(1.e-8)
            terms["path_relative_correction_rms"] = terms["path_mean_correction_rms"] / rms
            terms["path_accepted_steps"] = hr.new_tensor(diagnostics["accepted_steps"])
        return terms

    def _sr_reconstruction_loss(self, lr, base, hr, case_ids):
        """Supervise paired HR from deployment states, in physical image units.

        A full batch learns the clean endpoint from the deployment initial state.
        One eighth also learns from detached final and intermediate sampler states.
        Physical terms share fixed TRAIN variance normalization; dimensionless
        SSIM is added afterward. Neither the target nor per-outcome weights enter
        the main network's input state.
        """
        from .bridge import integrate_endpoint, sample_gaussian_bridge

        model, cfg = self.model, self.config
        train, sampling = cfg["train"], cfg["sampling"]
        if self.teacher is not None:
            raise ValueError("SR reconstruction supervises paired HR and does not use a teacher")
        steps = sampling["steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("SR reconstruction requires a positive deployment step count")
        weights = {"start": train.get("start_weight", 1.),
                   "rollout": train.get("rollout_weight", 1.),
                   "hf": train.get("hf_weight", .1),
                   "edge": train.get("edge_weight", .05),
                   "ssim": train.get("ssim_weight", 0.),
                   "mls": cfg.get("mls_reg_weight", 0.),
                   "bridge_aux": train.get("bridge_aux_weight", 0.)}
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("SR reconstruction weights must be finite and nonnegative")
        if weights["mls"] and (self.atlas is None or case_ids is None):
            raise ValueError("MLS reconstruction requires a TRAIN atlas and case IDs")

        condition = model.lr_encoder(lr)
        truth = model.normalize_haar(model.codec.encode(hr - base)).detach()
        band_scale = (model.band_std.detach() + 1.e-6 if model.use_band_whitening
                      else torch.ones_like(model.band_std)).to(hr)
        # Haar is orthonormal, so the average band variance is a pixel variance.
        # It is fixed by TRAIN statistics and is shared by every physical loss.
        normalization = band_scale.square().mean()
        with torch.no_grad():
            initial = model.initial_state(lr, base)
        start, _ = model(initial, condition, 0.)

        ssim_budget = max(1, len(lr) // 8)

        def components(prediction, target, image_base, image_hr, ids):
            coefficients = model.denormalize_haar(prediction)
            reconstructed = image_base + model.codec.decode(coefficients)
            error = reconstructed - image_hr
            pixel = error.square().mean()
            coefficient_error = coefficients - model.denormalize_haar(target)
            # Zero LL but retain the denominator of all sixteen channels.
            high_frequency = torch.cat((torch.zeros_like(coefficient_error[:, :1]),
                                        coefficient_error[:, 1:]), dim=1).square().mean()
            # Adjacent finite differences have operator norm two. Divide by two
            # before squaring so this term stays in physical pixel-MSE units.
            edge = .5 * ((error[..., 1:, :] - error[..., :-1, :]).div(2).square().mean()
                         + (error[..., 1:] - error[..., :-1]).div(2).square().mean())
            mls_terms = {}
            if weights["mls"] and hasattr(self.atlas, "reconstruction_terms"):
                mls_terms = self.atlas.reconstruction_terms(
                    prediction, target, ids, band_scale,
                    base_coefficients=model.codec.encode(image_base), reference_image=image_hr)
                mls = (mls_terms["local_mse"] + self.atlas.settings.get("neighbor_weight", 0.)
                       * mls_terms["neighbor_mse"])
            else:
                mls = (self.atlas.reconstruction_loss(prediction, target, ids, band_scale,
                                                     base_coefficients=model.codec.encode(image_base))
                       if weights["mls"] else pixel.new_zeros(()))
            raw = dict(pixel_mse=pixel, hf_mse=high_frequency, edge_mse=edge,
                       mls_reconstruction_mse=mls)
            raw.update({"mls_" + name: value for name, value in mls_terms.items()})
            combined = (pixel + weights["hf"] * high_frequency + weights["edge"] * edge
                        + weights["mls"] * mls) / normalization
            if weights["ssim"]:
                from .mri_metrics import mean_xy_ssim
                # Exact evaluation window/covariance convention, on a bounded
                # subset. Keep unclipped physical MSE above to correct outliers.
                count = min(len(prediction), ssim_budget)
                structural = 1 - mean_xy_ssim(reconstructed[:count].clamp(0, 1), image_hr[:count])
                combined = combined + weights["ssim"] * structural
                raw.update(ssim_loss=structural, ssim_items=pixel.new_tensor(count))
            return combined, raw

        start_loss, start_terms = components(start, truth, base, hr, case_ids)
        n = max(1, len(lr) // 8)
        subset_condition = [level[:n] for level in condition]
        detached_condition = [level.detach() for level in subset_condition]
        subset_ids = case_ids[:n] if case_ids is not None else None
        # The first and final states are already supervised separately. A real
        # intermediate grid point exists only when the deployment uses >=3 steps.
        intermediate_time = (int(torch.randint(1, steps - 1, (), device=lr.device)) / steps
                             if steps > 2 else None)
        intermediate_state = None

        def prefix_prediction(state, time):
            nonlocal intermediate_state
            if time == intermediate_time:
                # For Heun, the later call at this time is the accepted state,
                # overwriting the preceding predictor/corrector proposal.
                intermediate_state = state.detach()
            return model(state, detached_condition, time)[0]

        with torch.no_grad():
            final_state, final_time = integrate_endpoint(
                prefix_prediction, initial[:n].detach(), steps,
                sampler_type=sampling.get("sampler_type", model.sampler_type),
                solver=sampling.get("solver", model.solver),
                sigma_bridge=model.sigma_bridge,
                sigma_infer=sampling.get("inference_sigma"), return_prefix=True)
        final, _ = model(final_state, subset_condition, final_time)
        final_loss, final_terms = components(final, truth[:n], base[:n], hr[:n], subset_ids)
        rollout_loss = final_loss
        terms = {f"start_{name}": value for name, value in start_terms.items()}
        terms.update({f"rollout_{name}": value for name, value in final_terms.items()})
        if intermediate_state is not None:
            intermediate, _ = model(intermediate_state, subset_condition, intermediate_time)
            intermediate_loss, intermediate_terms = components(
                intermediate, truth[:n], base[:n], hr[:n], subset_ids)
            rollout_loss = .5 * (final_loss + intermediate_loss)
            terms.update({f"intermediate_{name}": value
                          for name, value in intermediate_terms.items()})
            terms["intermediate_time"] = hr.new_tensor(intermediate_time)
        loss = weights["start"] * start_loss + weights["rollout"] * rollout_loss
        if weights["bridge_aux"]:
            state, time, _ = sample_gaussian_bridge(
                truth, model.sigma_0, model.sigma_bridge, initial=initial,
                time_min=train.get("matching_time_min", 1.e-4))
            auxiliary, _ = model(state, condition, time)
            auxiliary_loss, auxiliary_terms = components(auxiliary, truth, base, hr, case_ids)
            loss = loss + weights["bridge_aux"] * auxiliary_loss
            terms.update({f"bridge_aux_{name}": value for name, value in auxiliary_terms.items()})
        terms.update(loss=loss, start_loss=start_loss, rollout_loss=rollout_loss,
                     endpoint_pixel_mse=start_terms["pixel_mse"],
                     loss_normalization=normalization, rollout_items=hr.new_tensor(n),
                     rollout_time=hr.new_tensor(final_time))
        return terms

    def _x1_loss(self, lr, base, hr, case_ids):
        from .bridge import sample_gaussian_bridge, integrate_endpoint, sample_induced_coupling

        model, cfg = self.model, self.config
        condition = model.lr_encoder(lr)
        truth = model.normalize_haar(model.codec.encode(hr-base))
        target, initial = truth, None
        if self.teacher is not None:
            self.teacher.eval()
            initial, target = sample_induced_coupling(self.teacher, lr, base,
                                cfg["straightening"].get("teacher_steps",8))
        state, t, _ = sample_gaussian_bridge(target, model.sigma_0,
                        0. if self.teacher is not None else model.sigma_bridge,
                        initial=initial, time_min=cfg["train"].get("matching_time_min",1.e-4))
        clean, _ = model(state, condition, t)
        error = clean-target
        epsilon = cfg["train"].get("charbonnier_eps",1.e-3)
        balanced = (error.square()+epsilon**2).sqrt().mean() - epsilon
        if cfg["train"].get("coefficient_loss", "charbonnier") == "l1":
            balanced = error.abs().mean()
        mse = F.mse_loss(base+model.codec.decode(model.denormalize_haar(clean)), hr)
        # Exactly 1/8 per rank (one item for tiny smoke batches). Prefix is detached.
        n = max(1, len(lr)//8)
        steps = cfg["train"].get("rollout_steps",3)
        detached_condition = [level[:n].detach() for level in condition]
        with torch.no_grad():
            initial = model.initial_state(lr[:n], base[:n])
            prefix, last_t = integrate_endpoint(lambda x,t: model(x,detached_condition,t)[0],
                                initial, steps, return_prefix=True, sigma_bridge=model.sigma_bridge)
        last, _ = model(prefix, [level[:n] for level in condition], last_t)
        rollout = F.mse_loss(base[:n]+model.codec.decode(model.denormalize_haar(last)),hr[:n])
        loss = (cfg["train"].get("coefficient_weight",1.)*balanced +
                cfg["train"].get("pixel_weight",1.)*mse +
                cfg["train"].get("rollout_weight",1.)*rollout)
        terms = dict(loss=loss, coefficient_loss=balanced, endpoint_pixel_mse=mse,
                     rollout_pixel_mse=rollout, rollout_items=hr.new_tensor(n))
        weight = cfg.get("mls_reg_weight",0.)
        if weight:
            if self.atlas is None or case_ids is None:
                raise ValueError("MLS regularization requires a TRAIN residual atlas and case IDs")
            normal = self.atlas.normal_loss(clean, truth.detach(), case_ids)
            terms.update(mls_normal_loss=normal, loss=loss+weight*normal)
        return terms


class X1HaarResidualNetwork(HaarResidualNetwork):
    """Single clean endpoint head; no direction embedding or reverse drift head."""
    predict_target = "x1"

    def __init__(self, widths, sigma_0=.03, sigma_bridge=.08, use_band_whitening=True,
                 multiscale_xy=True, sampler_type="ode", solver="euler"):
        super().__init__("haar_bridge", widths, "endpoint")
        self.direction, self.drift = None, None
        self.lr_encoder = LRConditionEncoder(widths, multiscale_xy)
        self.sigma_0, self.sigma_bridge = sigma_0, sigma_bridge
        self.use_band_whitening = use_band_whitening
        self.sampler_type, self.solver = sampler_type, solver
        self.register_buffer("band_mean", torch.zeros(16))
        self.register_buffer("band_std", torch.ones(16))
        self.register_buffer("band_stats_ready", torch.tensor(not use_band_whitening))

    @torch.no_grad()
    def set_band_statistics(self, payload):
        from .mri_data import _haar_statistics
        mean, std = _haar_statistics(self.band_mean.new_zeros(1,16,1,1,1),payload["mean"],payload["std"])
        self.band_mean.copy_(mean.flatten())
        self.band_std.copy_(std.flatten())
        self.band_stats_ready.fill_(True)

    def normalize_haar(self, value):
        if not bool(self.band_stats_ready):
            raise ValueError("TRAIN Haar statistics must be fitted before training or inference")
        if not self.use_band_whitening:
            return value
        return (value-self.band_mean.view(1,16,1,1,1))/(self.band_std.view(1,16,1,1,1)+1.e-6)

    def denormalize_haar(self, value):
        if not bool(self.band_stats_ready):
            raise ValueError("TRAIN Haar statistics must be fitted before training or inference")
        if not self.use_band_whitening:
            return value
        return value*(self.band_std.view(1,16,1,1,1)+1.e-6)+self.band_mean.view(1,16,1,1,1)

    def initial_state(self, lr, base, generator=None):
        shape = self.zero_state(lr,base).shape
        return self.sigma_0*torch.randn(shape, dtype=lr.dtype, device=lr.device, generator=generator)

    @torch.no_grad()
    def restore(self, lr, base, steps=8, sigma=None, generator=None, atlas=None, noise_steps=None,
                sampler_type=None, solver=None, initial_state=None):
        from .bridge import integrate_endpoint
        if atlas is not None:
            raise ValueError("x1 inference has no online atlas routing")
        state = self.initial_state(lr,base,generator) if initial_state is None else initial_state
        if state.shape != self.zero_state(lr,base).shape:
            raise ValueError("Initial state shape differs from the LR grid")
        condition = self.lr_encoder(lr)
        residual = integrate_endpoint(lambda x,t: self(x,condition,t)[0], state, steps,
                       sampler_type=sampler_type or self.sampler_type, solver=solver or self.solver,
                       sigma_bridge=self.sigma_bridge, sigma_infer=sigma,
                       generator=generator, noise_steps=noise_steps)
        return base+self.codec.decode(self.denormalize_haar(residual))


def make_haar_model(config):
    if config.get("predict_target") == "x1":
        return X1HaarResidualNetwork(config["model"]["widths"], config.get("sigma_0",.03),
                    config.get("sigma_bridge",.08), config.get("use_band_whitening",True),
                    config["model"].get("multiscale_xy",True),
                    config["sampling"].get("sampler_type",config.get("infer_sampler","ode")),
                    config["sampling"].get("solver","euler"))
    return HaarResidualNetwork(config["mode"], config["model"]["widths"],
                               config["sampling"].get("readout", "integral"), config.get("spatial_atlas"))
