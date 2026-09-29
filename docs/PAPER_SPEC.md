# TADE-EF paper specification

The submitted English manuscript and its `main.tex` are the only normative
method sources for this repository. Historical Chinese drafts and the legacy
38-dimensional implementation are not normative.

## Equation-to-module map

| Manuscript equations | Responsibility | Module |
|---|---|---|
| (1)-(4) | events, 20-ms windows, activity map, circular dilation | `candidates.py` |
| (5)-(6) | joint association cost and valid-motion criterion | `tracking.py` |
| (7)-(14) | Huber velocity fit, event alignment, alignment gain | `alignment.py` |
| (15)-(20) | NDFT, flatness, harmonic score, spatial phase | `spectral.py` |
| (21)-(23) | morphology and trajectory kinematics | `features.py` |
| (24) | fixed 36-dimensional segment vector | `schema.py`, `features.py` |
| (25)-(27) | centered log odds, accumulated evidence, identity lock | `evidence.py` |

## Execution contract

1. Candidate generation does not use frequency evidence.
2. Association uses the joint cost in Eq. (5), not IoU-priority fallback.
3. Motion compensation uses Huber regression and only current/past events.
4. The classifier input has exactly 36 features and contains no focal length,
   distance, or observation-count metadata.
5. TabPFN probabilities are consumed in chronological segment order.
6. Once a UAV or Non-UAV identity is locked, it is not changed.
7. Frame-level and track-level metrics are reported separately.

