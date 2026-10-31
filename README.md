# Trade-coupled commons

Code and supporting data for **Local allocation rules shape cascading collapse in trade-coupled commons**, by Tianyu Shi, Hanqing Qian and Chengyi Tu.

The model embeds local common-pool allocation games in wheat, maize, rice and soybean trade networks. This repository is self-contained; it does not require code or data from the authors' other papers.

## Install and regenerate figures

Python 3.11 or later is recommended. In a virtual environment:

```sh
python -m pip install -r requirements.txt
python reproduce.py figures
```

This uses the included stored simulation results and writes five main figures and ten Extended Data figures to `figs/`, as PDF and PNG. It does not rerun the simulations. To recompute summaries first, run `python reproduce.py summaries`.

## Files and inputs

- `ccsim2.py`: agent-based allocation, regeneration, trade, shocks, behaviour and mobility.
- `theory2.py`: stationary allocation, growth factors, node reduction and first-order criteria.
- `networks2.py`, `build_nets.py`: FAOSTAT harmonisation, mirror reconciliation, network construction and variants.
- `human.py`: published-game calibration and reliability analysis.
- `run2.py`: experiment families and deterministic seeds.
- `report2.py`, `extra_thresholds.py`: summaries, tables and source data.
- `figs2.py`, `figstyle.py`: all manuscript figure generation.
- `nets/`: processed networks and calibration samples.
- `results/`: stored experiment outputs, audit/calibration summaries and numerical tables.
- `source_data/`: per-panel/tabular supporting values.
- `tm_staples.csv.gz`, `qcl_staples.csv.gz`, `human_rounds.csv.gz`: fixed source extracts; see [DATA_SOURCES.md](DATA_SOURCES.md).

To rebuild all processed inputs from the supplied extracts:

```sh
python reproduce.py inputs
```

The extract files are included so the exact input snapshot remains available when upstream data change. The original extraction/download scripts were not part of the supplied manuscript package; this repository does not claim to regenerate those extracts automatically from today's upstream databases.

## Simulations

```sh
python reproduce.py quick --procs 2
python reproduce.py full --procs 4
```

`quick` samples a small number of jobs for E1 and E3 and writes separate `_quick.csv.gz` files. It is an installation check, not a reproduction of all reported estimates. `full` explicitly runs the production families and then regenerates summaries and figures; runtime depends strongly on hardware. The original manuscript records about 50,000 simulation jobs. No automatic full rerun occurs when importing this wrapper or choosing `figures`.

Individual families remain callable, for example `python run2.py E1 E3 --procs 4`. Use the same Python/NumPy stack for the closest numerical agreement; continuous-pool reduction diagnostics can be sensitive to roundoff perturbations on non-stationary trajectories; see VALIDATION.md for the observed discrepancies and limits.

## Validation and licences

See [VALIDATION.md](VALIDATION.md) for actual independent checks and their limits. The software is MIT licensed. Source data retain their third-party licences and attribution, described in [DATA_SOURCES.md](DATA_SOURCES.md); the MIT licence does not override them. No manuscript, review correspondence, private author records or credentials are included.

Use the repository's versioned release and accompanying Zenodo archival record when citing the software. `CITATION.cff` identifies the authors and software title.

## Fixed archival version

Version **v1.0.0**: [10.5281/zenodo.23838930](https://doi.org/10.5281/zenodo.23838930). The archived release corresponds to commit `85df235f61c8a9a6a285addf7edbf03e92eb4fd2`; all 123 archived files were checked against that fixed Git tree. The archival version is distinct from later documentation-only changes on the default branch.
