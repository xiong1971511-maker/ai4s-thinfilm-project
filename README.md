# AI4S Thin-Film Research Project

A course research project on multilayer dielectric thin films. The study uses the transfer matrix method (TMM) to generate spectra, trains multilayer perceptrons (MLPs) as surrogate models, and checks candidate rankings with full-pool TMM calculations.

## Paper

- [Final research article (PDF)](paper/AI4S_ThinFilm_Research_Article_Submission.pdf)
- [Editable article (DOCX)](paper/AI4S_ThinFilm_Research_Article_Submission.docx)

## Study at a glance

- Stack: Air/H/L/H/L/Glass at normal incidence, with fixed, non-absorbing refractive indices.
- Dataset: 5,000 four-layer thickness samples and 41 reflectance wavelengths from 400 to 800 nm.
- Split: fixed 4,000/500/500 train/validation/test partitions.
- Surrogate: 4 → 128 → 128 → 64 → 41 MLP; the Linear output model was selected by validation MSE.
- Candidate check: a fixed pool of 10,000 candidates was re-evaluated with TMM. Reported extrema apply only to this pool; they are not global optima over continuous thickness space.

## Reproducibility files

- Dataset: [data/thinfilm_dataset.npz](data/thinfilm_dataset.npz)
- Model and TMM implementation: [src/ai4s_thinfilm](src/ai4s_thinfilm)
- Selected Linear checkpoint: [models_convergence_linear/mlp_train_4000.pt](models_convergence_linear/mlp_train_4000.pt)
- Training metrics and histories: [outputs_convergence_linear](outputs_convergence_linear)
- Full-pool TMM audit: [outputs_convergence_linear/global_tmm_audit](outputs_convergence_linear/global_tmm_audit)
- Held-out spectrum used in Figure 3: [paper/heldout_test_case_row_3477_spectra.csv](paper/heldout_test_case_row_3477_spectra.csv)

## Run the checks

The pinned dependencies are listed in [requirements.txt](requirements.txt). In Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python scripts/check_tmm.py
python -m unittest discover -s tests
```

The TMM check covers the wavelength grid, Fresnel interface result, reflectance bounds, and energy conservation for the lossless model. The unit tests cover the dataset, candidate ranking, MLP, and TMM implementation.

## Scope and limitations

The model uses synthetic TMM data with fixed refractive indices, no absorption, and normal incidence. It does not model dispersion, polarization, manufacturing tolerances, or experimental measurement error. Candidate-design conclusions are limited to the saved 10,000-candidate pool and stated model settings.
