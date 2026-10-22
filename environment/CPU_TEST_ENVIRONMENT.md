# Independently installed CPU-check environment

This is a NEW validation environment created on 22 October 2026, not the historical training environment. A fresh Python 3.12 venv without system-site packages was populated from the official PyTorch CPU wheel index and PyPI. No historical environment was modified. The package's 28 synthetic CPU tests and `pip check` passed with isolated Python (`-I`) and CUDA hidden. No research image, prediction, training, inference, bootstrap or GPU execution was used by these tests.

The installed CPU torch build is 2.5.1+cpu. The headless OpenCV wheel 4.10.0.84 is a deliberate choice for this NEW CPU fixture environment; do not present its wheel suffix as a recovered historical pin. See requirements-cpu-checks.txt for the remaining exact installed dependencies.

Installation sequence (use a new environment, never the historical research one):

```bash
python3.12 -m venv .venv-cpu
.venv-cpu/bin/python -I -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
.venv-cpu/bin/python -I -m pip install --index-url https://pypi.org/simple -r environment/requirements-cpu-checks.txt
.venv-cpu/bin/python -I -m pip check
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/python -I -B -m unittest discover -s tests -v
```

These package choices and test invocation were exercised in the new venv. The lock file was recorded after installation; a second independent replay of every transitive pinned version was not performed. The venv and downloaded binaries are NOT bundled or uploaded. The requirements are CPU fixture requirements, not permission to run scientific models on CPU as a substitute for frozen CUDA experiments.

## Current research-interpreter resolution

The default research interpreter resolves several dependencies from the user's site-packages. When user site-packages are disabled, current torch/numpy versions differ and torchvision/timm/Mamba imports are not equivalent. See current_import_resolution_audit.csv. Neither resolution is silently substituted for historical experiment receipts. This is why a named interpreter alone is not an adequate complete environment specification.

causal-conv1d was not importable in the current default resolution, although mamba_ssm imported and its modules initialized on CPU. This does not prove or disprove a historical optional extension path or current CUDA forward/backward behavior. Full historical CUDA/Mamba installation and execution equivalence remain unverified here.
