# Data sources and attribution

## FAOSTAT extracts

`tm_staples.csv.gz` contains the staple-crop subset of FAOSTAT Detailed Trade Matrix; `qcl_staples.csv.gz` contains the corresponding Crops and Livestock Products production subset. The supplied manuscript package records access on 30 September 2026; that historical download date was not independently witnessed during this release audit.

Source: Food and Agriculture Organization of the United Nations (FAO), FAOSTAT, https://www.fao.org/faostat/en/#data/TM and https://www.fao.org/faostat/en/#data/QCL . FAO's [Statistical Database Terms of Use](https://www.fao.org/contact-us/terms/db-terms-of-use/en/) licence statistical datasets under CC BY 4.0 unless specific metadata state otherwise, with additional FAO terms. Credit FAO and acknowledge that this repository contains selected/transformed data. No FAO endorsement is implied. Raw units, harmonisation and filters are implemented in `networks2.py`; rice is converted to milled equivalent before reconciliation.

## Published commons-game extract

`human_rounds.csv.gz` is the supplied long-format extract from the public data accompanying Koster and colleagues (2025), *Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem*, Nature Communications 16, 2824, https://doi.org/10.1038/s41467-025-58043-7 .

Official repository: https://github.com/google-deepmind/sustainable_behavior . Its README's **License and disclaimer** section explicitly assigns Apache-2.0 to software and CC BY 4.0 to all other materials. This repository redistributes the data extract under that CC BY 4.0 source licence, with attribution to the original authors and Google DeepMind. It does not relicense the source data under MIT.

The upstream repository was inspected at commit `4f1a99a9d150f9fa6bad1a0f70f11c0673d46763` during this audit; this is not a claim that the manuscript's original extraction used that commit. Its notebook points to https://storage.googleapis.com/sustainable_behavior/sustainable_behavior.csv .

The extract contains `game`, `launch`, `cond`, `round`, `pool`, `player`, `offer` and `action`: experimental identifiers and game measurements. It does not contain names, contact information or credentials. The source wide-format observations have been selected and reshaped to player-round format. `human.py` further filters valid offers/actions and rounds below 30 for the reported calibration: 680 games, 2,720 player-games and 51,353 player-rounds. These are the analysed subset, not the total participant count of the source study. During this audit, 6,448 reconstructed player-rounds from the original wide CSV matched the extract identifiers and pool/offer/action values exactly. The complete 214 MB upstream CSV was not downloaded or compared row by row. The original wide-to-long extraction script was not supplied; this release preserves the extract without inventing its provenance.

## Derived material

`nets/`, `results/`, `source_data/` and generated `figs/` contain transformations and simulations from these inputs. Credit the present software authors and the relevant original data source when reusing derived data. The source-specific attribution and licence terms remain applicable to incorporated source material. The present authors' original numerical outputs are offered under CC BY 4.0; Python software is MIT.
