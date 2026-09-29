R1.2 candidate comparison — 28 September 2026

**Deferred by the user before any GPU run started.** Finish discussing Reviewers
2 and 3 before executing the revision plan. The worker was stopped while waiting
for GPU 1, and its temporary reservation was released. Baseline reproduction was
not stopped. The opt-in implementation, CPU checks and proposed configuration
remain preserved; they are not completed scene experiments.

The user wants the eventual study to test removal and replacement and choose the
better supported option. The exact replacement was recovered from the earlier
[fix-options note](R1_2_FIX_OPTIONS.md):

\[
d=n^\top(x-p),\qquad P^*(x)=\exp[-d^2/(2h^2)],\qquad
g^*(x)=\nabla_xP^*(x)=-\frac{d}{h^2}P^*(x)n.
\]

Here n is a unit world-space normal, h>0, and p and n are fixed while taking the
local derivative. This directly supervises a probability-like field. The selected
replacement does not introduce a learned global SDF or tangential envelope.

| Candidate | Scalar target | Gradient loss |
| --- | --- | --- |
| `normal_off` | Radial Gaussian about the associated support point | None |
| `signed_plane` | The exact P* above | Squared vector error against g*, multiplied by h² for coordinate scaling; zero target retained on the surface |

Both pilots use probability L1 weight 3000 and RGB L1 weight 1. Replacement uses
weight 100 for the h-scaled vector error. Both use the same local query and
nearest-support association rule, world-space normals, expected rendered depth,
support filtering, field bandwidth, Gaussian initialization, seed and optimizer
recipe. Queries include anchors and Gaussian offsets with standard deviation h;
h is 0.05 times the initial training-camera scene radius. The same fixed nearest
anchor supplies a query's scalar and gradient target. Consistency is local to
that association; the piecewise target is not asserted to be globally smooth
across nearest-anchor boundaries.

This new common support/query pipeline differs from the historical launcher.
Consequently, this pilot compares two repaired candidates; it is not the final
one-factor ablation of the original normal term. R1.2 still owns the appropriate
matched gradient-loss control for the eventual selected method, shared with
R1.6. Initial COLMAP reconstruction is shared; whether it used training-only
observations is unknown. These pilots cannot settle that provenance issue.

The prepared, deferred scene comparison would use Bonsai and Garden, their frozen 60-degree test
lists, seed 0, images_4 at native resolution, a 500k Gaussian cap and a fixed
12,000-iteration final checkpoint. This is a diagnostic budget, not the final
30k paper protocol. Two 320-iteration low-resolution smoke runs first cross CPF
activation and verify checkpoint rendering and nonzero applied CPF pull. No such
run has started. The proposed execution device was local physical GPU 1; the
temporary scheduler reservation has been removed after the user's deferral.

Select the replacement if its mean of the two scene PSNRs exceeds removal by
more than 0.05 dB; otherwise prefer removal. Report all PSNR/SSIM/LPIPS triples
and tradeoffs. This rule was saved before either result existed. The choice is
provisional because it uses one seed and two diagnostic scenes; it does not
establish general robustness. Do not reuse held-out pilot selection as evidence
of unbiased generalization.

Every run retains the PLY, full Gaussian checkpoint, CPF weights, CPF optimizer
and field state, loss/mechanism diagnostics, exact arguments, logs and fresh
per-view metrics. Evaluation uses the frozen test list, native rendering, 8-bit
RGB/GT PNG values, repository PSNR/SSIM, and lpipsPyTorch VGG v0.1. Each metric
triple comes from the same fixed final checkpoint.

Evidence root on local host `user` (historical alias `hust10`):

[/data1/zhangao/outputs/GS-LSS-R1.2-compare-20260928-142236](/data1/zhangao/outputs/GS-LSS-R1.2-compare-20260928-142236)

- [Protocol and frozen code hashes](/data1/zhangao/outputs/GS-LSS-R1.2-compare-20260928-142236/protocol.json)
- [Dataset and split hashes](/data1/zhangao/outputs/GS-LSS-R1.2-compare-20260928-142236/dataset_manifest.json)
- [Live status](/data1/zhangao/outputs/GS-LSS-R1.2-compare-20260928-142236/status.json)
- [Worker log](/data1/zhangao/outputs/GS-LSS-R1.2-compare-20260928-142236/worker.log)
- [Analytic/gradient checks](/data1/zhangao/outputs/GS-LSS-R1.2-compare-20260928-142236/target_checks.log)

Seven CPU checks passed: analytic versus automatic derivatives, both sides and
the peak, normal-flip/tangential invariance, tilted-plane world normals,
nearest-support assignment, removal of the derivative penalty, finite nonzero
network updates from vector-gradient supervision, and bandwidth validation
(several properties share a test). Python syntax checks passed. Scene experiments
are deferred, with zero GPU runs started; consult live status. No winning objective has been selected and
the submitted manuscript and its numerical tables have not been changed.
