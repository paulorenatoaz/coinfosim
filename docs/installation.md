# Installation

## End users: install from PyPI

```bash
python -m pip install coinfosim
```

Requires Python 3.10+. This installs the `coinfosim` console script and every runtime dependency (NumPy, SciPy, scikit-learn, pandas, matplotlib, Typer, etc.) — no separate `pip install -r requirements.txt` step, and no need to clone this repository. Raw datasets are **not** part of the wheel; they are downloaded and hash-verified on first use (see [`docs/datasets.md`](datasets.md)).

Verify the install:

```bash
coinfosim --version
coinfosim scenario list
coinfosim doctor
```

## Optional extras

```bash
python -m pip install "coinfosim[dev]"      # pytest, ruff, mypy, build, twine
python -m pip install "coinfosim[legacy]"   # deprecated Google Sheets/Drive integration
```

Neither extra is required for `coinfosim scenario run ...`.

## Developers: editable install from a source checkout

```bash
git clone https://github.com/paulorenatoaz/coinfosim.git
cd coinfosim
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
python -m pip install -e ".[dev]"
```

An editable install still resolves datasets the same way as a wheel install (platform cache, download, or `--data-dir`); the repository's `data/raw/<dataset>` files provide an additional automatic fallback (step 6 of the resolution order) only when running from a source checkout, so a developer working from a clone with those files already present does not need to download anything to run a scenario.

## Building and checking a distribution locally

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Verify the wheel excludes raw datasets and includes the packaged catalog:

```bash
python -m zipfile -l dist/coinfosim-*-py3-none-any.whl | grep resources/datasets.json   # must appear
python -m zipfile -l dist/coinfosim-*-py3-none-any.whl | grep -i data/raw                 # must be empty
```

## Uninstalling

```bash
python -m pip uninstall coinfosim
```

The dataset cache under your platform's user data directory (e.g. `~/.local/share/coinfosim` on Linux) is not removed automatically; delete it manually if you no longer need the cached datasets.
