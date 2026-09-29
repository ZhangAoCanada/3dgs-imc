R1.2: concrete repair options and decision sequence — 28 September 2026

The user accepted the interpretation that R1.2 concerns incompatibility between
CPF probability-value targets and normal-gradient targets. Sampling a point
from a Gaussian primitive and differentiating CPF at that point is a separate
operation. This note elaborates the [R1 plan](/data1/zhangao/repos/3dgs-imc-codex/writings/tvcg-major-revision-2026-09-28/R1_REVISION_PLAN.md).
The user wants the eventual plan to try both removal and replacement and use
the better supported option. They explicitly selected the earlier signed-normal
displacement equations below for replacement, then clarified that execution
must wait until the Reviewer 2 and 3 discussions are complete. The queued worker
was stopped before any GPU run began. No winning loss has been selected.
The analytic illustration is not a scene-reconstruction result.

The [candidate-comparison record](R1_2_COMPARISON.md) holds the exact implemented
targets, bounded pilot protocol, checks and live execution evidence.

The repair principle is to define what the scalar field means, then derive any
gradient target from that same definition. Removing an incompatible gradient
loss also removes this specific conflict. We do not need to project
Gaussian-center updates onto normals to answer this comment.

**A: retain the probability target and remove the normal-gradient loss**

Keep the existing probability and configured color supervision. Set the normal
loss weight to zero. For the manuscript's isotropic target about a support point
p, the network is still trained to predict

    P*(x) = exp(-||x-p||^2 / (2 sigma^2)).

CPF remains a differentiable network. Its input gradient remains available to
the Gaussian update: removing direct supervision on that gradient does not
remove the gradient or disable CPF. When the predicted field approximates the
target, its gradient is expected to reflect the surrounding values; finite
samples and small value errors alone do not guarantee accurate derivatives.
That is one reason to inspect the gradients explicitly.

This removes the specific simultaneous requirement to fit a radial scalar
target while aligning its derivative with a fixed normal. It does not resolve
every possible support-quality or multi-point target issue. It also removes
normal supervision of the field, so its effect on useful geometry is unknown
until measured.

This is the first recommended diagnostic because it isolates the disputed term
and directly supplies R1.6's normal-loss ablation. Keep color/probability weights,
support, seeds, iteration, and the Gaussian recipe fixed in the paired contrast.
The [current loss code](/data1/zhangao/repos/3dgs-imc-codex/scene_experimental_branch3/gaussian_model.py:825)
hardcodes the normal weight; changing configuration alone must not be assumed
to disable it in that version. An eventual implementation should expose an
explicit, logged loss weight and verify its effect.

If derivative supervision is useful while retaining this isotropic target,
its consistent target is `grad P* = -(x-p) P*/sigma^2`. That is radial derivative
supervision, not normal alignment, and should be named accordingly. Multiple
support points still require an explicit aggregation/association rule if the
claim is about the derivative of one exact global target field.

**B: the user-selected replacement — Gaussian of signed normal displacement**

Let p be a support point, n a unit normal, and x a query. In a small neighborhood
of a locally planar surface, its signed normal displacement is

    d(x) = n^T (x-p).

Use a probability target high at the local surface and falling on both sides:

    P*(x) = exp(-d(x)^2 / (2 h^2)).

With p and n held fixed, its derivative is

    g*(x) = grad P*(x) = -(d(x)/h^2) P*(x) n.

For positive d, the gradient points along -n; for negative d it points along +n.
At d=0 it is zero. Thus both sides move toward the surface under gradient ascent
of this local support score. Reversing normal orientation changes both d and n
in sign and leaves this scalar target and its derivative unchanged.

The normal is retained as information about the surface's orientation. It no
longer means that probability must increase along +n everywhere. A field peaked
at a surface has different derivative semantics from a signed-distance field.

One possible training objective is

    L = wP mean |P_theta(x)-P*(x)|
        + wg mean ||grad_x P_theta(x)-g*(x)||^2
        + wc L_color.

All coordinates and gradient targets must use the same normalization; weights
need scaling accordingly. A vector loss permits a zero target gradient at the
surface. If a direction-only loss is used instead, mask near-zero target
gradients because they have no defined unit direction. This proposal changes
the probability target and the gradient loss together. Changing only the sign
of the original normal loss is insufficient for arbitrary isotropic samples.

This exact pair of equations is the replacement to test against removal. Do not
silently substitute a learned SDF or the finite-patch extension below. The signed
quantity is the local displacement d; the learned target P* remains an unsigned
probability-like support score. Supervise the full derived vector g*, including
its zero at d=0, rather than retaining the old fixed-normal cosine target.

The infinite plane above is a local model. Using it everywhere
would assign high probability far from observed support along that plane.
An alternative discussed earlier was a finite patch with tangential falloff. It
is retained here as background, not the replacement selected for this test. With
`u = (I-n n^T)(x-p)`, use

    K(x) = exp(-d^2/(2 h_n^2) - ||u||^2/(2 h_t^2)),
    grad K(x) = -K(x) [d n/h_n^2 + u/h_t^2],

where h_t > h_n makes the patch broader along the surface than across it.
The tangential derivative is then nonzero away from the patch center. The loss
must use this complete derivative, not a purely normal direction. This is the
intuition behind the earlier anisotropic-kernel proposal. It concerns CPF
support targets and does not prescribe the shape of the rendered 3DGS primitives.

For several points, first specify how the patches combine into one scalar
target at each location and then differentiate that target. Independently
assigning incompatible values/derivatives from overlapping patches would
reintroduce a related problem. This detail and neighborhood sizes require a
prototype; they have not been finalized. Noisy normals, curvature and patch
boundaries remain practical limitations despite target consistency.

**What the decision experiment should establish**

Compare the existing objective, A, and the B prototype. First use analytic
planes and spheres with known geometry, inspecting field values and derivatives
on both sides and near zero-gradient locations. A sphere tests curved geometry;
a locally planar patch approximation is not an exact global sphere field.
Then test the selected candidates on Bonsai and Garden with matched recipes.
Include independently referenced depth where verified data is available.

This R1.2 study also owns the normal-loss comparison used in the R1.6 response.
If B is retained with a gradient loss, include its matched gradient-loss-off
control with the same scalar target and other losses; original versus B alone
does not isolate that term. If A is chosen, use the existing-loss versus A
removal comparison. The shared comparison is not a separate R1.6 training job.

The toy check establishes target/implementation consistency; it cannot establish
reconstruction quality. The scene pilot establishes whether the proposed change
preserves or improves useful behavior. Choose using both pieces of evidence.
Prefer A if normal supervision is unnecessary. Pursue B if retaining normal
information is useful enough to justify the additional field definition and
implementation work. Neither result is known yet.

Keeping the existing loss as an explicitly competing heuristic is a possible
interpretation, but calling it regularization does not prove its targets agree.
It would need empirical justification and would abandon the exact-consistency
claim rather than repair the targets. It is not the primary repair proposed here.

After selecting a change, align equations, code and descriptions. A new loss
creates a revised trained method: regenerate affected GS-LSS results attributed
to it, while reusing compatible unchanged baselines. The small diagnostics
precede that decision. Do not relabel existing checkpoints as trained with the
new objective.

**SIREN motivation clarified in discussion**

The user explained that the normal term was inspired by SIREN, with the intent
of improving continuity of the learned surface. This records the user's design
motivation; it is not a claim that SIREN establishes compatibility with CPF.

The [SIREN paper, Section 4.2, Eq. (6) in the linked arXiv version](https://arxiv.org/pdf/2006.09661)
learns a signed-distance field, with zero values and normal supervision on the
surface, and an Eikonal constraint over the domain. The
[official loss implementation](https://github.com/vsitzmann/siren/blob/master/loss_functions.py#L192-L215)
masks normal supervision to surface samples; its
[point-cloud loader](https://github.com/vsitzmann/siren/blob/master/dataio.py#L379-L399)
marks off-surface samples with a sentinel. These sources were checked on
28 September 2026. Our current code instead concatenates support and additional
sample locations, copies the support normals, and applies its cosine loss
without that surface-only mask (gaussian_model.py:710, 819, 845).

For a plane, the signed distance d(z)=z has positive derivative on both sides.
The CPF target exp(-z^2/(2 sigma^2)) has opposite derivative signs on opposite
sides and a zero derivative at its maximum. Thus the normal constraint's meaning
depends on the scalar field it supervises. Applying it only at CPF support peaks
would not by itself solve the conflict either. Smoothness of a neural scalar
function, accuracy of its reconstructed surface, and compatibility of its loss
targets are different questions.

A possible SIREN-based variant of replacement option B is to learn d_theta with
appropriate SDF supervision and derive P_theta=exp(-d_theta^2/(2 h^2)). Normal
alignment would then constrain grad d_theta, while Gaussian guidance would use
grad P_theta=-(d_theta/h^2) P_theta grad d_theta. This is a proposed model change,
not a reinterpretation of the existing direct-probability checkpoints; reliable
oriented support and the field's behavior beyond observed regions would still
need investigation. No replacement or new training was selected in this discussion.

**Local signed coordinate versus a learned signed-distance field**

The earlier local-patch replacement uses d=n dot (x-p) as a signed normal
coordinate inside a probability target. It still uses a squared term in the
exponential and can keep a network that predicts probability directly. Its
consistent target gradient contains the factor -d, producing the appropriate
opposite directions on the two sides of the patch. The signed coordinate alone
does not repair the loss if a fixed +n probability-gradient target is retained.
For a finite patch, include the tangential derivative as specified above.

Learning d_theta as an SDF and then deriving P_theta from it is a separate
parameterization. A local plane projection is not automatically a globally
valid signed distance to the scene surface, especially for curved, incomplete
or open surfaces. The discussion favored investigating the smaller local-patch
change before adopting a global SDF representation; that is an experimental
priority, not evidence that it performs better or a user decision to adopt it.
The normal-off option also remains available as the minimal alternative.
