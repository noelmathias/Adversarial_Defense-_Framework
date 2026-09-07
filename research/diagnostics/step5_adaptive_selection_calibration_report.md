# Step 5: Adaptive-selection calibration analysis

**Source run:** `results/adaptive_fixed_policy_benchmark_20260905T115712Z`  
**Scope:** read-only reconstruction from the completed Step 4 episode records. No
benchmark configuration, controller, reward, or result artifact was changed.

## Reconstruction and integrity basis

For each of the 1,200 adaptive cases (10 seeds x 120 cases), the corresponding
fixed-light, fixed-medium, and fixed-strong rows have the same seed, case,
image, attack, epsilon, adversarial payload, context, and arm-specific defense
and post-risk seeds. When adaptive selected an arm, its realized reward matched
that fixed arm's row. Thus the fixed-arm rows supply matched realized
counterfactual rewards for the arm-choice diagnostic.

The original global integrity audit passed: matched cases, fixed-arm identity,
one adaptive update per case, fixed-policy LinTS bypass, shared
verification/risk/reward path, and the expected 4,800 evaluations.

The oracle arm is the fixed arm with the highest realized reward for the exact
matched case. There were no ties among the three fixed-arm rewards.

## Overall calibration

| Quantity | Result |
|---|---:|
| Adaptive decisions matching matched oracle | 462 / 1,200 (38.5%) |
| Mean realized adaptive reward | 0.34732 |
| Mean matched-oracle reward | 0.42228 |
| Mean oracle regret | 0.07495 |
| Median oracle regret | 0.03000 |
| Positive-regret decisions | 738 / 1,200 (61.5%) |
| Total realized oracle regret | 89.94407 |
| Light oracle cases | 1,121 (93.4%) |
| Medium oracle cases | 74 (6.2%) |
| Strong oracle cases | 5 (0.4%) |

### Adaptive choice versus matched oracle

Rows are adaptive choices; columns are the unique matched oracle arm.

| Chosen arm | Light oracle | Medium oracle | Strong oracle | Total |
|---|---:|---:|---:|---:|
| Light | 439 | 27 | 1 | 467 |
| Medium | 373 | 23 | 4 | 400 |
| Strong | 309 | 24 | 0 | 333 |
| Total | 1,121 | 74 | 5 | 1,200 |

Adaptive selected a non-oracle Medium or Strong arm on 682 cases (56.8%). It
also selected Light on 28 of the 79 non-Light-oracle cases. The main error is
therefore not failure to ever select Light; it is frequent selection of
Medium/Strong when Light was the matched best action.

## Regime-level calibration

| Regime | Oracle-match rate | Mean regret | Adaptive L/M/S selections | Oracle L/M/S cases |
|---|---:|---:|---:|---:|
| FGSM 0.01 | 41.5% | 0.05501 | 86 / 65 / 49 | 190 / 9 / 1 |
| FGSM 0.03 | 37.5% | 0.08065 | 76 / 61 / 63 | 187 / 11 / 2 |
| FGSM 0.05 | 41.5% | 0.05500 | 78 / 70 / 52 | 184 / 16 / 0 |
| PGD 0.01 | 37.0% | 0.06986 | 73 / 73 / 54 | 193 / 6 / 1 |
| PGD 0.03 | 36.0% | 0.08630 | 73 / 67 / 60 | 183 / 17 / 0 |
| PGD 0.05 | 37.5% | 0.10290 | 81 / 64 / 55 | 184 / 15 / 1 |

Calibration is poorer under PGD (36.8% matched-oracle; mean regret 0.08635)
than FGSM (40.2%; 0.06355). By epsilon, match rates are 39.3%, 36.8%, and
39.5% for 0.01, 0.03, and 0.05; mean regret is highest at epsilon 0.03
(0.08348), followed by 0.05 (0.07895).

## Learning trajectory

Each block pools the corresponding 20 manifest positions across all ten seeds.
The fixed manifest becomes materially harder late in the sequence, so falling
reward or increasing regret alone must not be interpreted as posterior
deterioration.

| Cases | Adaptive L/M/S | Oracle L/M/S | Match | Mean reward | Mean regret |
|---|---:|---:|---:|---:|---:|
| 1-20 | 86 / 65 / 49 | 190 / 9 / 1 | 41.5% | 0.41529 | 0.05501 |
| 21-40 | 76 / 61 / 63 | 187 / 11 / 2 | 37.5% | 0.38188 | 0.08065 |
| 41-60 | 78 / 70 / 52 | 184 / 16 / 0 | 41.5% | 0.39047 | 0.05500 |
| 61-80 | 73 / 73 / 54 | 193 / 6 / 1 | 37.0% | 0.39189 | 0.06986 |
| 81-100 | 73 / 67 / 60 | 183 / 17 / 0 | 36.0% | 0.29650 | 0.08630 |
| 101-120 | 81 / 64 / 55 | 184 / 15 / 1 | 37.5% | 0.20790 | 0.10290 |

There is no evidence of improving calibration: oracle matching stays around
36-42%, and Strong remains 24.5-31.5% of choices even though it is the oracle
in at most two cases per block. This is descriptive rather than a causal
estimate of learning rate because time is confounded with the fixed regime
order.

## Context, cost, and effectiveness

The selector changes its mix by regime, but not in a way aligned with the
matched oracle. For example, Strong selection is 31.5% for FGSM 0.03 and 30.0%
for PGD 0.03, although Strong is oracle in only 2/200 and 0/200 respective
cases. Across risk quartiles, Strong selection changes only from 25.7% in the
lowest to 29.3% in the highest; mean regret rises from 0.03511 to 0.16993.

| Selected arm | Cases | Light better than selected | Selected better than Light | Mean selected-minus-Light reward |
|---|---:|---:|---:|---:|
| Medium | 400 | 377 (94.3%) | 23 (5.8%) | -0.02664 |
| Strong | 333 | 327 (98.2%) | 6 (1.8%) | -0.13194 |

The regret attributable to the 377 Medium choices where Light was better is
22.1686 total (mean 0.0588). The corresponding 327 Strong choices contribute
56.3431 total (mean 0.1723). Medium/Strong do occasionally beat Light (29
combined chosen cases), so a literal "always Light" rule is not a per-case
oracle. But the observed selector does not identify those exceptions reliably.

The existing reward intentionally penalizes nominal cost: Light, Medium, and
Strong have normalized cost terms of 0.0333, 0.3333, and 1.0, weighted by 0.1.
Adaptive's mean cost penalty is about 0.04016 versus Light's 0.00333; this
0.03682 difference accounts for about 81% of its 0.04550 mean-reward deficit
to Light. Cost is not the sole explanation: Light also had better aggregate
recovery, accuracy, post-risk, and risk reduction.

## Interpretation and limitation

The strongest supported explanation is **selection miscalibration under this
finite, ordered 120-case benchmark**: the posterior-updating controller
continues to allocate substantial probability to costly arms whose matched
outcomes are usually inferior. The evidence is compatible with insufficient
learning horizon, weak context-to-arm-outcome signal, and/or persistent
Thompson-sampling exploration. It cannot distinguish those mechanisms
causally. It does not support an implementation fault, a claim that the reward
is erroneous, or a claim that CUSUM caused the result.

No new performance experiment is required to support the current defensible
negative conclusion. If mechanism attribution is essential, the smallest
justified follow-up would be a separately preregistered, counterbalanced-order
learning-curve diagnostic with unchanged controller/reward/arms, explicitly
testing whether calibration improves when the same matched regimes are not
confounded with time. It must be presented as diagnosis, not as a tuning or
attempt to overturn Step 4.
