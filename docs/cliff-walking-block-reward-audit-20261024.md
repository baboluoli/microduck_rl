# CUDA reward audit before the block pilot

RTX A6000; untouched source 500; development seeds 20–22; all 72 first episodes including failures. Values are actual weighted, dt-integrated point contributions. Distributions and every episode are retained in the audit JSON and episodes JSON.

| Term | Mean | Min | P25 | Median | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|
| track_linear_velocity | 21.4314 | 4.9893 | 19.9180 | 22.6313 | 24.6257 | 32.4030 |
| track_angular_velocity | 1.4895 | 0.1774 | 0.3496 | 0.4695 | 0.9725 | 12.0167 |
| upright | 9.5245 | 1.6834 | 8.1899 | 9.3712 | 11.1953 | 21.7191 |
| pose | 0.3839 | 0.1263 | 0.3365 | 0.3857 | 0.4419 | 0.5954 |
| body_ang_vel | -1.7540 | -3.2977 | -2.2081 | -1.6817 | -1.3233 | -0.3348 |
| angular_momentum | -0.0000 | -0.0000 | -0.0000 | -0.0000 | -0.0000 | -0.0000 |
| dof_pos_limits | -0.6254 | -2.2262 | -0.8670 | -0.4823 | -0.3141 | -0.0313 |
| action_rate_l2 | -5.9747 | -13.6613 | -6.9188 | -5.9765 | -5.1595 | -1.0043 |
| air_time | 1.4742 | 0.3000 | 0.9000 | 1.2600 | 1.9500 | 5.3400 |
| foot_clearance | -0.1558 | -0.6709 | -0.1801 | -0.1440 | -0.1165 | -0.0403 |
| foot_swing_height | -0.2358 | -0.6153 | -0.2898 | -0.2301 | -0.1631 | -0.0350 |
| foot_slip | -0.0515 | -0.1167 | -0.0635 | -0.0549 | -0.0421 | -0.0029 |
| self_collisions | -6.9503 | -43.7800 | -5.9750 | -3.0700 | -1.1500 | -0.0800 |
| head_pose_tracking | 10.2806 | 2.4429 | 8.4898 | 9.8295 | 11.8421 | 17.9874 |
| body_pose_tracking | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| head_pose_bias | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| upright_progress | -0.0688 | -0.1208 | -0.1126 | -0.1061 | -0.0104 | -0.0000 |
| servo_impact | -0.0150 | -0.6287 | -0.0007 | 0.0000 | 0.0000 | 0.0000 |
| head_impact | -0.0027 | -0.0078 | -0.0043 | -0.0020 | -0.0011 | 0.0000 |
| trunk_impact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| servo_acc_spike | -0.1856 | -0.4537 | -0.2317 | -0.1730 | -0.1291 | -0.0241 |
| servo_stall | -0.2097 | -0.5410 | -0.2675 | -0.1760 | -0.1313 | -0.0240 |
| cliff_landing | 19.4444 | 0.0000 | 8.0000 | 24.0000 | 24.0000 | 40.0000 |
| positive_dense | 45.5837 | 11.1513 | 40.0469 | 45.1099 | 52.8552 | 74.8738 |
| negative_dense | -17.2289 | -53.6900 | -19.0219 | -12.8267 | -9.5440 | -1.7674 |
| block_bonus | 19.4444 | 0.0000 | 8.0000 | 24.0000 | 24.0000 | 40.0000 |
| final_bonus | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Penalty-only mean (13 negative-weight terms): -16.1606; range [-53.0296, -1.7416]. Negative dense totals above also include negative samples of signed upright-progress shaping.

All penalty totals are nonpositive. All actual block/final payouts match independently reconstructed eligibility and the prescribed point amounts. Synthetic tests exercise the strict 40-point payout and simultaneous 48-point final-step payout, which the source episodes do not reach.

Decision before PPO updates: keep the proposed five 8-point block bonuses and 40-point final bonus. The average block credit is 42.7% of positive dense reward mass; this is a substantial reachable signal. This audit does not establish optimal weights or predict learning. No optimizer, threshold, physics or other reward changes were made.
