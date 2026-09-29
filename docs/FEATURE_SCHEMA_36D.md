# Fixed 36-dimensional feature schema

The order below is part of the model artifact contract.

## Frequency and spatial phase (11)

1. `alignment_gain_mean`
2. `alignment_gain_max`
3. `peak_frequency_mean_hz`
4. `peak_frequency_std_hz`
5. `peak_frequency_cv`
6. `spectral_flatness_mean`
7. `spectral_flatness_std`
8. `harmonic_score_mean`
9. `harmonic_score_max`
10. `spatial_phase_concentration_mean`
11. `spatial_phase_concentration_max`

## Morphology (8)

12. `elongation_mean`
13. `elongation_std`
14. `elongation_cv_mean`
15. `elongation_cv_max`
16. `box_area_cv_mean`
17. `box_area_cv_max`
18. `box_aspect_cv_mean`
19. `box_aspect_cv_max`

## Motion (11)

20. `path_length_px`
21. `endpoint_displacement_px`
22. `straightness`
23. `speed_mean_px_per_s`
24. `speed_max_px_per_s`
25. `local_speed_mean_window_mean`
26. `local_speed_std_window_mean`
27. `local_acceleration_mean_window_mean`
28. `local_acceleration_std_window_mean`
29. `local_jerk_mean_window_mean`
30. `local_curvature_mean_window_mean`

## Observation quality (6)

31. `duration_ms`
32. `roi_event_count_mean`
33. `box_area_mean_px2`
34. `box_area_std_px2`
35. `box_aspect_mean`
36. `box_aspect_std`

Relative to the legacy model, `focal_length`, `distance`,
`observation_count`, and `acceleration_max_px_per_s2_max` are excluded.
`local_speed_mean_window_mean` and
`local_acceleration_std_window_mean` are added.

