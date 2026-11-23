# Road-combination geometry: implementation decisions

This document records the first geometry-focused implementation pass. It does
not claim that all ACR stages are verified; no original game PAKs or generated
stage reports were available in this source-only review.

## Implemented in this pass

- Decode `FInterpCurveVector` input keys, interpolation modes, loop state and
  loop-key offset.
- Evaluate linear, constant and cubic interpolation using Unreal's key-interval
  tangent scaling. Cubic segments are converted to Bezier form and adaptively
  subdivided with a conservative control-polygon length bound and a flatness
  tolerance.
- Reject discontinuous constant segments rather than connecting them with an
  invented diagonal road segment.
- Reject manual joins that exceed the configured kink, grade or width limits;
  remove the old 180-degree placement fallback.
- Reject nearly parallel boundary crossings instead of clamping their small
  normal component and producing unbounded longitudinal deformation.
- Calculate vertical shear so the resulting grade is matched per transformed
  horizontal arclength, accounting for the horizontal speed change introduced
  by the lateral shear.
- Add regression tests for spline semantics, join derivatives and manual
  sequence rejection.

## Join shear mathematics

Let `e` be the tile-boundary tangent and `n` its selected normal. A road unit
heading at angle `a` is `u = sin(a)e + cos(a)n`. The existing compact-support
profile is `g(d) = d(1-d/D)^2` for `0 <= d < D`, and zero after `D`.
Its maximum is `4D/27`; it has zero displacement at the boundary and zero
first derivative at the outer support edge.

The boundary-parallel shear coefficient changes the tangent angle from the
source angle to the mean angle `a_m`. This also changes horizontal speed:
`|dXY_new/ds_source| = cos(a_source)/cos(a_m)`. Therefore the vertical shear
must match grade after this speed change. With source grades `q_in` and
`q_out`, and `q_m = (q_in + q_out)/2`, the implemented coefficients are:

- next side: `Az_out = q_m/cos(a_m) - q_out/cos(a_out)`
- previous side: `Az_in = q_in/cos(a_in) - q_m/cos(a_m)`

This makes both transformed one-sided grades equal to `q_m` at the join when
grade is measured as `dz / horizontal_distance`. Property tests exercise
random crossing orientations and grades.

The current displacement profile is C1 but not C2 at its outer support edge.
A higher-order polynomial is not substituted automatically: candidate profiles
can introduce stronger counter-shear or other undesirable derivatives and must
be compared against source-stage geometry before selection.

## Deliberately deferred

The following require representative original/generated ACR data or in-game
validation before safe implementation:

- Numeric maximum deformation budgets for vertical and boundary-parallel shifts.
- Matching left/right road border splines to centerline stations and preserving
  crossfall/banking in generated route splines.
- Separating terrain-only seam blending from road meshes and road collision.
  The current source has path-name classifications, but their correctness and
  relationship to ACR's baked collision meshes need validation on real stages.
- Road-wide collision sampling and surface classification using actual
  material/owner mappings.
- Tree/rock/barrier broad-phase and narrow-phase clearance using real
  BodySetup/AggGeom and instance transforms.
- Grid-corner supercover traversal and stage-specific grid calibration.

No vertical- or boundary-parallel-deformation budget has been invented in
this pass. The boundary-normal projection floor (currently 0.2, configurable)
is a provisional numerical guard that caps the inverse projection factor at 5;
it has not been calibrated against ACR stage availability. The existing join
limits also remain provisional and should be calibrated against Hafren South
and Elatia source geometry and measured in-game results.
