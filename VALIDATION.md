# Independent validation of this package

Audit date: 31 October 2026. The actual software environment is recorded in `validation_environment.json`. Python was 3.13.5; numerical packages and thread settings are recorded separately from the original manuscript's Python 3.11/Linux production environment.

## Checks performed

1. Rebuilt all networks from the supplied FAOSTAT extracts and recalculated the human calibration from the supplied published-game extract. All 36 network NPZ files / 180 arrays were **exactly equal** to the supplied processed inputs. All six existing calibration NPY arrays matched within 1e-12.
2. Recomputed `net_audit.json`, `human_summary.json`, `numbers.json`, `table1.json` and `extra_thresholds.json` from the supplied inputs and stored simulations. No numeric differences exceeded 1e-9 relative / 1e-11 absolute tolerance. Reliability was exactly 0.6408540923692713 and Spearman–Brown reliability 0.7811225816476169, matching stored values.
3. Actually regenerated five main and ten Extended Data figures, and visually compared every figure with the manuscript package. Numerical trends and panels agree. The long-iteration labels in Extended Data Fig. 4 were corrected from “converged” to “3,000 max.” because the iteration can reach its limit without converging.
4. Independently reran eight E1 jobs: four crops × equal/resource-contingent sharing, top-five shocks, replicate zero, across fourteen coupling values each. This gives 112 rows with 68 numeric fields each. Equal-sharing runs matched the stored results at 1e-8 relative / 1e-10 absolute tolerance. Resource-contingent runs matched all fields except four continuous reduced-pool error diagnostics, detailed below. Depletion/cascade outcomes and reduction classification counts matched.

## Continuous reduction diagnostics: observed limit

The four exceptions are `red_interp_conv_absx`, `red_interp_h300_absx`, `red_exact_conv_absx` and `red_exact_h300_absx`. The maximum original/current difference was **0.00020664395994**, for maize at coupling zero in the 300-round interpolation variant: 3.84% relative to that small stored error, or 0.0207 percentage points on the unit-pool scale.

Controlled comparisons held the network, agents, seeds and agent-model window fixed. Mathematically equivalent summation or a single first-round perturbation of about 1e-16 can alter non-stationary reduced-map trajectories and shift the continuous error by 1e-5–1.8e-4. One wheat perturbation reproduced a historical error to about 1.25e-13. All depletion labels and numerical stopping flags remained unchanged in 92 control/variant records. Individual transient pool differences can be substantial; this is not a claim that every trajectory is numerically identical.

The archived and rebuilt network arrays were compared exactly, and a fourteen-row maize rerun using the archived NPZ files reproduced the current rerun exactly. Network reconstruction differences are therefore excluded here. The specific historical roundoff trigger is **not uniquely established**; it would be inaccurate to attribute it conclusively to a particular CPU, BLAS implementation or old script version.

Stored historical simulations are preserved. No global tolerance was relaxed to hide the four exceptions, and no historical errors were overwritten to force agreement. The sampled scientific classifications are supported by actual reruns; invariance of every untested production job has not been demonstrated.

## Stopping flag and method descriptions

In `theory2.reduced`, `converged` means successive iterates differ by less than 1e-10, with a maximum of 3,000 rounds. The legacy output field `cycle` means the run did not converge and the last two iterates differ by more than 1e-6. It does **not** estimate a period or prove periodic dynamics. The source docstring, manuscript stopping description and non-stationarity statistics were corrected accordingly; the algorithm and stored counts are unchanged. The explicit mathematical period-three step-rule example is separate from this Boolean flag.

`human.py` splits rounds 0–14 and 15–29. The manuscript's previous odd/even label was corrected without changing the estimator or values. Its fixed effects are at the player-game level.

## Scope

The full original set of about 50,000 production jobs was not independently rerun. Regeneration from stored outputs and the representative reruns are distinct checks. The original wide-to-long data extraction script was not supplied; the fixed extract and upstream source/attribution are documented in DATA_SOURCES.md. The source data were spot-checked against the public upstream observations, not reconstructed in full during this audit.
