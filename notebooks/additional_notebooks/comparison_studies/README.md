# Comparison-studies validation

Validates BEWARE-modelled runup against six published field/model studies (Storlazzi et al.
2018 — Roi-Namur; Hoeke et al. 2021 — Tarawa, Nanumea, Funafuti, Nukulaelae, Nui), and documents
why the main pipeline's `BEWARE_Database_extended_v4.nc` needed a wider companion table for this
comparison specifically.

Run in this order:

1. **`022_extend_beware_by_eta_H0_H0L0_wide.py`** — re-runs the same interpolation/extrapolation
   algorithm as `notebooks/022_extend_beware_by_eta_H0_H0L0.py`, but with `H0`/`H0L0`/`eta0`
   widened to the raw BEWARE grid's own bounds (`H0` up to 5.0 m instead of 3.2 m), because the
   main pipeline's `v4.nc` was sized only to the scenario pipeline's own input range and truncates
   the larger wave heights (up to 5.9 m) reported by these validation studies. Produces
   `BEWARE_Database_extended_v4_wide.nc`, which reproduces `v4.nc` bit-for-bit on their shared
   grid while additionally covering these validation studies' larger wave heights.
2. **`031_extract_from_BEWARE_comparison_studies.py`** — runs the matching for all six studies
   against `v4_wide.nc` and reports RMSE/bias/MAE against the published values.
