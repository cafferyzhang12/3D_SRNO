"""Build the Chinese multi-seed TEST report from a completed frozen experiment matrix."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .mri_compare import paired_seed_comparison
from .runtime import fingerprint


METRICS = ("all_psnr_db", "mean_slice_ssim", "all_mse", "foreground_psnr_db", "hfen", "edge_mse", "z_difference_mse")
CONTRASTS = (("bridge", "direct"), ("mls", "pca"), ("mls_geometry", "mls_shuffled"))
HIST = "5281446_original"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_counts(summary):
    cases = summary["cases"]
    return (len(cases), sum(c["metrics"]["evaluated_slices"] for c in cases),
            sum(c["metrics"]["pixel_count"] for c in cases))


def verify_metric_totals(summary):
    counts = case_counts(summary)
    if (counts != (84, 14991, 3929800704) or counts[1] != summary["total_slices"]
            or summary.get("pixel_count", counts[2]) != counts[2]):
        raise ValueError("Expected the original 84-case, 14,991-slice, 3,929,800,704-voxel MRI cohort")
    rows = [c["metrics"] for c in summary["cases"]]
    mse = sum(row["squared_error_sum"] for row in rows) / counts[2]
    n_ssim = sum(row["ssim_slice_count"] for row in rows)
    if n_ssim != counts[1] or any(r["ssim_slice_count"] != r["evaluated_slices"] for r in rows):
        raise ValueError("Every retained original slice must contribute one SSIM value")
    ssim = sum(row["mean_slice_ssim"] * row["ssim_slice_count"] for row in rows) / n_ssim
    for metric, value in (("all_mse", mse), ("all_psnr_db", -10 * math.log10(mse)), ("mean_slice_ssim", ssim)):
        if not math.isclose(value, summary[metric], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError(f"Stored {metric} differs from pooled case evidence")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def delta_row(candidate, reference, training_seed, pairs, draws):
    diff = paired_seed_comparison(pairs, resamples=draws)
    row = dict(candidate=candidate, reference=reference, training_seed=training_seed)
    for key in ("pooled_psnr_db", "pooled_mse", "mean_slice_ssim"):
        stat = diff[key]
        row[key] = stat["difference"]
        row[key + "_low95"], row[key + "_high95"] = stat["ci95"]
    return row


def f(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


def with_sd(value, spread, digits):
    return f(value, digits) if spread is None else f"{f(value, digits)} ± {f(spread, digits)}"


def conclusion(row):
    lo, hi = row["pooled_psnr_db_low95"], row["pooled_psnr_db_high95"]
    verdict = "成对区间支持提升" if lo > 0 else ("成对区间支持下降" if hi < 0 else "成对区间跨越或触及零")
    return (f"PSNR 差 {row['pooled_psnr_db']:+.4f} dB，病例配对 95% 区间 [{lo:+.4f}, {hi:+.4f}]，{verdict}。"
            f"SSIM 差 {row['mean_slice_ssim']:+.6f}，95% 区间 "
            f"[{row['mean_slice_ssim_low95']:+.6f}, {row['mean_slice_ssim_high95']:+.6f}]。")


def compute_profile(summary, method, group):
    """Logical per-tile/per-sample evaluations, independent of vectorized tile batching."""
    steps = summary["steps"]
    nfe = summary.get("network_calls_per_tile_sample", summary.get("drift_network_calls_per_tile_sample"))
    guidance = summary.get("geometry_guidance") or summary.get("momentum_geometry")
    input_text = "real LR bilinear lift on the HR grid"
    condition, atlas, atlas_inputs, backward = 0, 0, 0, 0
    if group == "bilinear":
        nfe = 0
    elif group == "external":
        input_text = ("single native LR slice repeated to RGB, channel-mean readout" if method == "swinir"
                      else "single LR slice, nearest-neighbour x4 lift and fixed [-1,1] scaling")
    elif group in {"new", "isolation"}:
        input_text = "native LR plus HR-grid bilinear base; all 16 Haar XY residual bands"
        condition = 1
        atlas = int(method in {"mean", "pca", "mls", "mls_geometry", "mls_shuffled", "matched_mls", "matched_pca"})
        atlas_inputs = atlas
    elif guidance:
        # GSBM uses t=(i+1)/steps at interior times. Momentum uses t=i/steps.
        active = sum(i / steps > guidance["start"] for i in range(1, steps))
        if summary.get("momentum_geometry"):
            atlas, atlas_inputs, backward = 1 + active, 1 + 3 * active, 2 * active
        else:
            atlas = 1 + active * (2 if guidance.get("normal_strength", 0) else 1)
            atlas_inputs, backward = atlas, active
    if nfe is None and "momentum" in summary["inference_variant"]:
        nfe = steps
    if nfe is None:
        raise ValueError(f"Missing reconstruction-network evaluation count for {method}")
    return dict(input_description=input_text, network_calls_per_tile_sample=nfe,
                lr_condition_encoder_calls=condition, atlas_encoder_calls=atlas, atlas_encoder_inputs=atlas_inputs,
                guidance_backward_calls=backward, total_neural_forwards_per_tile_sample=nfe + condition + atlas)


def build(matrix_path, historical_path, output, resamples=10000, independent_root=None):
    historical = read(historical_path)
    matrix = read(matrix_path)
    verify_metric_totals(historical)
    if (round(historical["all_psnr_db"], 4), round(historical["mean_slice_ssim"], 6)) != (30.6781, .907021):
        raise ValueError("The historical summary must match job 5281446 epoch 38")
    grouped, runs, labels, all_cases, independent = defaultdict(dict), [], set(), [], []
    matched = defaultdict(dict)
    data_identity = None
    for record in matrix["runs"]:
        label, method, seed, group = (record[k] for k in ("label", "method", "training_seed", "group"))
        if group not in {"new", "historical", "external", "bilinear", "isolation"}:
            raise ValueError(f"Unexpected comparison group {group}")
        if label in labels or method == HIST or seed in grouped[method]:
            raise ValueError(f"Duplicate label or method/seed in frozen matrix: {label}")
        labels.add(label)
        path = Path(record["output"]) / "summary.json"
        s = read(path)
        freeze_dir = Path(record["freeze_dir"])
        freeze = read(freeze_dir / "freeze.json")
        selected = freeze["selected"]
        if ("freeze_sha256" in record and fingerprint(freeze_dir / "freeze.json") != record["freeze_sha256"]
                or "validation_sha256" in record and selected["summary_sha256"] != record["validation_sha256"]):
            raise ValueError(f"Frozen VAL evidence changed for {label}")
        if (s.get("split") != "test" or s["subset"] != historical["subset"]
                or read(freeze_dir / "manifest.json") != s["subset"]
                or selected["checkpoint_sha256"] != s["checkpoint_sha256"]
                or not math.isclose(selected["pooled_psnr"], record["validation_psnr"], rel_tol=1e-10)):
            raise ValueError(f"Frozen model, validation score, or TEST cohort differs for {label}")
        for key in ("seed", "steps", "samples", "predictor"):
            if selected["inference"][key] != s[key]:
                raise ValueError(f"Frozen {key} differs from evaluation for {label}")
        inference = selected["inference"]
        for key in ("drift_sampler", "readout"):
            if s["predictor"] != "bilinear" and key in inference and s.get(key) != inference[key]:
                raise ValueError(f"Frozen {key} differs from evaluation for {label}")
        if "inference_sigma" in inference and inference["inference_sigma"] != s["sampling_sigma"]:
            raise ValueError(f"Frozen sampling noise differs for {label}")
        if "disable_guidance" in inference and inference["disable_guidance"] == bool(s.get("geometry_guidance")):
            raise ValueError(f"Frozen online guidance differs for {label}")
        verify_metric_totals(s)
        grouped[method][seed] = s
        ext = s.get("external", {})
        rows = s["cases"]
        identity = {case["case_id"]: case["input_fingerprint"] for case in rows}
        if data_identity is not None and identity != data_identity:
            raise ValueError("Frozen TEST runs reference different inputs")
        data_identity = identity
        if group in {"new", "isolation"} and s.get("training_seed", seed) != seed:
            raise ValueError(f"Training seed differs from frozen matrix for {label}")
        for case in rows:
            m = case["metrics"]
            all_cases.append(dict(method=method, label=label, training_seed=seed, case_id=case["case_id"],
                                  slices=m["evaluated_slices"], voxels=m["pixel_count"],
                                  **{key: m.get(key) for key in METRICS}))
        if independent_root and record.get("confirm_archive"):
            separate = Path(independent_root) / label / "summary.json"
            held_out = read(separate)
            if (held_out["split"] != "independent_archive" or held_out["checkpoint_sha256"] != s["checkpoint_sha256"]
                    or held_out["freeze_sha256"] != fingerprint(freeze_dir / "freeze.json")):
                raise ValueError(f"Mismatched independent archive confirmation for {label}")
            independent.append(dict(label=label, seed=seed, cohort_sha256=held_out["cohort_sha256"],
                                    cases=len(held_out["cases"]), slices=held_out["total_slices"],
                                    voxels=held_out["pixel_count"], psnr=held_out["all_psnr_db"],
                                    ssim=held_out["mean_slice_ssim"], source=str(separate)))
        arm = record.get("arm", {})
        for role in record.get("roles", []):
            if role in {"matched_pca", "matched_mls"}:
                if seed in matched[role] or arm.get("variant") != role.removeprefix("matched_"):
                    raise ValueError("Missing or duplicate matched geometry role")
                matched[role][seed] = (arm, s)
        runs.append(dict(method=method, group=group, training_seed=seed, label=label,
                         validation_psnr=selected["pooled_psnr"], **{k: s.get(k) for k in METRICS},
                         dimensionality=s.get("input_dimensionality", "2D" if ext or group == "bilinear" else "5-slice"),
                         pretraining=s.get("pretraining", ext.get("pretraining")),
                         **compute_profile(s, method, group),
                         roles=",".join(record.get("roles", [])),
                         local_dimension=arm.get("dim"), geometry_in_path=arm.get("path"),
                         endpoint_readout=arm.get("readout"),
                         case_evaluation_seconds=sum(c["evaluation_seconds"] for c in rows),
                         cases=len(rows), slices=s["total_slices"], voxels=s["pixel_count"],
                         summary=str(path), summary_sha256=fingerprint(path), freeze_dir=str(freeze_dir),
                         checkpoint_sha256=s["checkpoint_sha256"]))

    methods, intervals = [], []
    for method, summaries in grouped.items():
        group_rows = [r for r in runs if r["method"] == method]
        group = group_rows[0]["group"]
        if any(r["group"] != group for r in group_rows):
            raise ValueError(f"Mixed comparison groups for {method}")
        if group in {"new", "isolation"} and set(summaries) != {42, 43, 44}:
            raise ValueError(f"Expected three frozen seeds 42, 43, 44 for {method}")
        if len({r["checkpoint_sha256"] for r in group_rows}) < len(group_rows):
            raise ValueError(f"Repeated checkpoint among purported training seeds of {method}")
        if group == "isolation":
            continue  # Retained in runs.csv, cases.csv and the prespecified matched contrast.
        row = dict(method=method, group=group, training_seeds=len(summaries),
                   validation_psnr=mean(r["validation_psnr"] for r in group_rows))
        for metric in METRICS:
            vals = [r[metric] for r in group_rows]
            row[metric] = mean(vals) if all(v is not None for v in vals) else None
        row["psnr_seed_sd"] = stdev(r["all_psnr_db"] for r in group_rows) if len(summaries) > 1 else None
        row["ssim_seed_sd"] = stdev(r["mean_slice_ssim"] for r in group_rows) if len(summaries) > 1 else None
        paired = delta_row(method, HIST, "all", [(historical, s) for s in summaries.values()], resamples)
        intervals.append(paired)
        row.update({"vs_5281446_" + k: v for k, v in paired.items()
                    if k not in ("reference", "candidate", "training_seed")})
        methods.append(row)
    by_method = {row["method"]: row for row in methods}
    methods.sort(key=lambda r: -r["all_psnr_db"])
    for i, row in enumerate(methods, 1):
        row["rank"] = i
    eligible = [r for r in methods if r["group"] == "new" and r["method"] != "mls_shuffled"]
    # Keep manifest order for equal validation scores. No TEST-dependent tie breaking.
    best = max(eligible, key=lambda r: (r["validation_psnr"], -list(grouped).index(r["method"])))
    if matrix.get("champion_method", best["method"]) != best["method"]:
        raise ValueError("Frozen champion differs from the validation-only selection")
    contrasts = []
    extra = [("mls", m) for m in ("direct", "bridge", "mean") if m in grouped]
    extra += [("mls_geometry", "mls")]
    for cand, ref in (*CONTRASTS, *extra):
        if cand not in grouped or ref not in grouped or set(grouped[cand]) != set(grouped[ref]):
            raise ValueError(f"Incomplete same-seed contrast {cand} versus {ref}")
        pairs = [(grouped[ref][s], grouped[cand][s]) for s in sorted(grouped[cand])]
        for s, pair in zip(sorted(grouped[cand]), pairs, strict=True):
            intervals.append(delta_row(cand, ref, s, [pair], resamples))
        row = delta_row(cand, ref, "all", pairs, resamples)
        intervals.append(row)
        contrasts.append(row)
    isolated = None
    if matched:
        if any(set(matched[role]) != {42, 43, 44} for role in ("matched_mls", "matched_pca")):
            raise ValueError("Matched geometry comparison requires three paired training seeds")
        pairs, dimensions = [], set()
        for seed in (42, 43, 44):
            pca, reference = matched["matched_pca"][seed]
            mls, candidate = matched["matched_mls"][seed]
            if (pca != {**mls, "variant": "pca"} or pca["path"] or pca["shuffle"]
                    or pca["query"] != "endpoint" or pca["readout"] != "endpoint" or pca["reference"] != "identity"):
                raise ValueError("Matched PCA and MLS differ beyond surface order")
            dimensions.add(pca["dim"])
            pair = (reference, candidate)
            pairs.append(pair)
            intervals.append(delta_row("matched_mls", "matched_pca", seed, [pair], resamples))
        if len(dimensions) != 1:
            raise ValueError("Matched dimension must remain fixed across training seeds")
        isolated = delta_row("matched_mls", "matched_pca", "all", pairs, resamples)
        intervals.append(isolated)
        matched_dim, = dimensions
    external = [r for r in methods if r["group"] == "external"]
    for ext in external:
        name = ext["method"]
        external_runs = grouped[name]
        if len(external_runs) != 1:
            raise ValueError("Expected one published pretrained model per external method")
        reference = next(iter(external_runs.values()))
        for cand in dict.fromkeys(("mls_geometry", best["method"])):
            pairs = [(reference, s) for s in grouped[cand].values()]
            intervals.append(delta_row(cand, name, "all", pairs, resamples))
    full = by_method["mls_geometry"]
    ext_rank = 1 + sum(r["all_psnr_db"] > full["all_psnr_db"] for r in external)
    best_diff = next(r for r in intervals if r["candidate"] == best["method"] and r["reference"] == HIST)
    lines = ["# 配对 MRI XY×4、Z×1 正式测试", "",
             f"1. 按平均整病例 VAL PSNR 选定 **{best['method']}**。相对第 38 轮历史模型，{conclusion(best_diff)}",
             f"2. 无图册桥相对同骨干直接回归，{conclusion(contrasts[0])}",
             (f"3. 固定局部维数 {matched_dim}、相同路径和查询后，二阶 MLS 相对局部 PCA，{conclusion(isolated)}"
              if isolated else f"3. 验证选定的 MLS 配置相对局部 PCA 配置，{conclusion(contrasts[1])}"),
             f"4. 真实协方差方向相对同谱随机方向，{conclusion(contrasts[2])}",
             (f"5. 完整模型 mls_geometry 与 {len(external)} 个已评估外部方法按 TEST PSNR 排序，"
              f"位列 **第 {ext_rank}/{1 + len(external)}**。"
              if external else "5. 当前矩阵缺少同协议外部方法，外部排名待确认。"),
             "", "## 冻结配置的整病例成绩", "",
             "历史参考为作业 5281446 的第 38 轮。固定清单含 84 例、14,991 张切片、3,929,800,704 个体素。"
             "候选方法按 VAL 冻结后再评分。每个训练种子先计算全体素 MSE 与 PSNR、逐切片平均 SSIM，"
             "表中按方法对种子取等权算术平均。更高的 PSNR 和 SSIM、更低的误差表示更好的结果。",
             "", "| TEST 排名 | 方法 | 种子数 | VAL PSNR | TEST PSNR ± 种子 SD | ΔPSNR 95% CI | SSIM ± 种子 SD | ΔSSIM 95% CI | MSE | 前景 PSNR | HFEN | 连续 Z 差分 MSE |",
             "|---:|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|"]
    for r in methods:
        lines.append(f"| {r['rank']} | {r['method']} | {r['training_seeds']} | {f(r['validation_psnr'],4)} | "
                     f"{with_sd(r['all_psnr_db'],r['psnr_seed_sd'],4)} | "
                     f"[{r['vs_5281446_pooled_psnr_db_low95']:+.4f}, {r['vs_5281446_pooled_psnr_db_high95']:+.4f}] | "
                     f"{with_sd(r['mean_slice_ssim'],r['ssim_seed_sd'],6)} | "
                     f"[{r['vs_5281446_mean_slice_ssim_low95']:+.6f}, {r['vs_5281446_mean_slice_ssim_high95']:+.6f}] | {f(r['all_mse'],9)} | "
                     f"{f(r['foreground_psnr_db'],4)} | {f(r['hfen'])} | {f(r['z_difference_mse'],9)} |")
    lines.extend(["", f"表中 Δ 以作业 5281446 原始 TEST 结果为参照。95% 区间采用 {resamples:,} 次病例配对自助重采样，随机种子为 42。"
                  "同一抽样在全部训练种子和两种方法间复用，区间描述固定训练种子下的病例抽样变异。"
                  "种子间标准差单列，单种子方法留空。各方法配置分别由 VAL 选择，"
                  "比较 MLS 与 PCA 的机制时还需结合相同局部维数的验证消融。", "",
                  "[各方法汇总及相对历史模型的区间](methods.csv) · [逐病例指标](cases.csv) · [逐种子得分、输入、预训练、网络调用、评估耗时及运行记录](runs.csv) · "
                  "[逐种子和跨种子配对差值](paired.csv)", "",
                  f"历史 summary 的 SHA-256 为 `{fingerprint(historical_path)}`。冻结矩阵 SHA-256 为 `{fingerprint(matrix_path)}`。"])
    lines += ["", "## 同种子比较", "", "| 候选 | 对照 | PSNR 差为正的种子 | PSNR 差范围（dB） |", "|---|---|---:|---:|"]
    comparisons = [*CONTRASTS, *extra]
    if isolated:
        comparisons.append(("matched_mls", "matched_pca"))
    for cand, ref in comparisons:
        differences = [r["pooled_psnr_db"] for r in intervals
                       if r["candidate"] == cand and r["reference"] == ref and r["training_seed"] != "all"]
        lines.append(f"| {cand} | {ref} | {sum(x > 0 for x in differences)}/{len(differences)} | "
                     f"{min(differences):+.4f} 至 {max(differences):+.4f} |")
    if matrix.get("matched_validation"):
        lines += ["", "## 固定局部维数的 MLS 与 PCA 验证消融", "",
                  "两臂均使用训练种子 42、干净端点查询、单位参考协方差和相同路径。下表为整病例 VAL 上 MLS 减 PCA。"
                  "TEST 主表比较各自在 VAL 选定的最终配置。", "",
                  "| 局部维数 | ΔPSNR（dB，95% CI） | ΔSSIM（95% CI） |", "|---:|---|---|"]
        for check in matrix["matched_validation"]:
            psnr, ssim = check["pooled_psnr_db"], check["mean_slice_ssim"]
            lines.append(f"| {check['dim']} | {psnr['difference']:+.4f} "
                         f"[{psnr['ci95'][0]:+.4f}, {psnr['ci95'][1]:+.4f}] | {ssim['difference']:+.6f} "
                         f"[{ssim['ci95'][0]:+.6f}, {ssim['ci95'][1]:+.6f}] |")
    diagnostics = matrix.get("sampler_validation")
    if diagnostics:
        lines += ["", "## 原第 38 轮权重的整病例 VAL", "",
                  "| 采样臂 | 漂移 NFE | PSNR | SSIM | ΔPSNR 相对 plain | ΔSSIM 相对 plain |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in diagnostics["ranking"]:
            lines.append(f"| {row['arm']} | {row['nfe']} | {row['psnr']:.4f} | {row['ssim']:.6f} | "
                         f"{row['delta_psnr']:+.4f} | {row['delta_ssim']:+.6f} |")
        lines += ["", "| 区域 | 元素数 | 〈e,g〉 | ΔMSE | 最优修正倍率 |", "|---|---:|---:|---:|---:|"]
        for region, values in diagnostics["correction"].items():
            lines.append(f"| {region} | {values['count']} | {values['dot']:.6g} | "
                         f"{f(values['delta_mse'],9)} | {f(values['optimal_change_multiplier'],4)} |")
        chart = diagnostics["chart"]
        lines += ["", f"LR 前三图对 HR 最优图的召回为 {chart['oracle_recalled']:.4f}，"
                  f"截断前的前三图概率质量为 {chart['top3_mass']:.4f}。"
                  f"LR 先验加权拟合误差相对 HR 最优图的平均差为 {chart['lr_prior_weighted_gap']:.6g}。"
                  f"全图引导诊断为“{diagnostics['correction']['whole']['interpretation']}”。"
                  "[分区、逐更新路由与采样器证据](sampler_validation.json)"]
    if matrix.get("development"):
        lines += ["", "## 训练路径和网络消融的整例 VAL", "",
                  "下表使用相同训练种子和固定局部维数。相对路径修正 RMS、有效近邻数和二次拟合回退率"
                  "来自入选权重所在 epoch 的 TRAIN 日志。[全部训练与 VAL 记录](development.csv)", "",
                  "| 查询 | 路径几何 | 路由打乱 | 路径相对修正 RMS | 有效近邻 | 二次回退率 | VAL 边缘 MSE | VAL PSNR | VAL SSIM |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in matrix["development"]:
            if row["stage"] == "controls":
                lines.append(f"| {row['query']} | {row['path']} | {row['shuffle']} | "
                             f"{f(row['path_relative_correction_rms'])} | {f(row['effective_neighbors'],2)} | "
                             f"{f(row['quadratic_fallback_fraction'],4)} | {f(row['val_edge_mse'],9)} | "
                             f"{f(row['val_psnr'],4)} | {f(row['val_ssim'])} |")
    lines += ["", "## 计算口径", "",
              "runs.csv 分列每次采样中每个窗口的重建网络、LR 条件编码器、图册编码器前向调用，以及在线引导的反向求导次数。"
              "动量 MLS 在一次编码器调用内批量编码三个状态，atlas_encoder_inputs 计入三次输入。"
              "这些是逐窗口的逻辑调用次数。窗口批处理改变实际设备执行批数。"
              "case_evaluation_seconds 为各病例耗时之和，含读取、重建、融合和评分。"]
    if independent:
        if len({row["cohort_sha256"] for row in independent}) != 1:
            raise ValueError("Independent confirmations used different case cohorts")
        lines += ["", "## 单独留出的档案病例", "",
                  f"同一数据档案的预留病例共 {independent[0]['cases']} 例，跨中心来源未经确认。"
                  f"VAL 选定模型在这些病例上的跨种子平均 PSNR 为 {mean(r['psnr'] for r in independent):.4f} dB，"
                  f"SSIM 为 {mean(r['ssim'] for r in independent):.6f}。[独立确认记录](independent.csv)"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "methods.csv", methods)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "cases.csv", all_cases)
    write_csv(output / "paired.csv", intervals)
    if diagnostics:
        (output / "sampler_validation.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf8")
    if matrix.get("development"):
        write_csv(output / "development.csv", matrix["development"])
    if independent:
        write_csv(output / "independent.csv", independent)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("matrix", "historical", "output"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--independent-root")
    args = parser.parse_args()
    build(args.matrix, args.historical, args.output, independent_root=args.independent_root)


if __name__ == "__main__":
    main()
