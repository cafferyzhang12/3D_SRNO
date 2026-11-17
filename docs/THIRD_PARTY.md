# Research provenance and upstream licenses

This isolated project implements a paired-restoration, fixed-coupling Gaussian
variational adaptation of conditional stochastic optimal control and neural
bridge matching. It is not a verbatim reproduction of upstream unpaired GSBM,
an IPF solver, or a proof of globally optimal endpoint coupling.

## Generalized Schrödinger Bridge Matching

Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos A.
Theodorou, Ricky T. Q. Chen. *Generalized Schrödinger Bridge Matching*, ICLR 2024.

- Paper: https://arxiv.org/abs/2310.02233
- Authors' implementation:
  https://github.com/facebookresearch/generalized-schrodinger-bridge-matching
- Inspected revision: `a8ab5b500dcea8b1c0df84822188f69758a08442`.
- Upstream is primarily CC BY-NC 4.0. Its README lists separately licensed
  components originating from deepgsb (Apache 2.0) and guided-diffusion (MIT).
- Source inspection concerned Gaussian conditional paths, CondSOC objective,
  and drift matching. The local PyTorch implementation deliberately avoids
  introducing the upstream Lightning training loop into the existing MRI I/O.

Changes include fixed observed LR/HR pairing, full-resolution local 3D paths,
bounded endpoint-zero variance and interior-mean spline optimization, voxelwise
observation masks, and train-only feature geometry. Path optimization adds a
feature-metric kinetic term alongside the reference control and state costs.
This is an extension/restriction of the reference objective and must be reported
as such; matching a finite variational family does not establish global SB
optimality or exact marginal satisfaction after finite neural training.

## Metric Flow Matching

Kacper Kapuśniak et al. *Metric Flow Matching for Smooth Interpolations on the
Data Manifold*, NeurIPS 2024.

- Paper: https://arxiv.org/abs/2405.14780
- Authors' implementation: https://github.com/kkapusniak/metric-flow-matching
- Inspected revision: `e44e03ec6207d2b111c1d51a03dcc53a98f87b95`.
- License: MIT, copyright (c) 2024 Kacper Kapusniak.
- Local adaptation: local second-moment / RBF diagonal metric in a learned 3D
  representation, plus a separate RBF support potential. The representation
  and centers are learned from training HR only. The adapted geometry is not
  the previous invertible-coordinate pullback and requires no named manifold,
  segmentation, skeleton, tree, or anatomical labels.

The local metric is a feature-space approximation. It neither identifies a
unique true data manifold nor guarantees preservation of spatial topology.
Cross-modality generality describes the parameterization and learning process;
MRI-to-EM zero-shot transfer is an untested empirical question.

The upstream license texts are included with the documentation. Keep the
applicable license notices with redistributed adaptations. [GSBM CC BY-NC 4.0](licenses/GSBM_CC_BY_NC_4.txt) · [MFM MIT](licenses/MFM_MIT.txt)

The single-chart stochastic benchmark uses the staged training idea from
*Branched Schrödinger Bridge Matching* and trains a stochastic drift for its own
fixed-coupling benchmark. The inspected code revision was
`249d34e9942bf0682c3dd3119b0ae6a886e53d26`. [Paper](https://arxiv.org/abs/2506.09007) · [Authors' repository](https://github.com/sophtang/BranchSBM)

The momentum variant uses the conditioned process described in *Momentum
Multi-Marginal Schrödinger Bridge Matching*. [Method and inspected revision](MOMENTUM.md)
