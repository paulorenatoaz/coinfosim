# Releasing CoInfoSim

## One-time external setup (repository owner only)

PyPI/TestPyPI publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/) (OIDC) — no long-lived API token is stored anywhere in this repository. Before the first release, the repository owner must register a Trusted Publisher on each index:

| Field | Value |
|---|---|
| Owner | `paulorenatoaz` |
| Repository | `coinfosim` |
| Workflow filename (TestPyPI) | `publish-testpypi.yml` |
| Workflow filename (PyPI) | `publish-pypi.yml` |
| Environment name (TestPyPI) | `testpypi` |
| Environment name (PyPI) | `pypi` |
| Project name | `coinfosim` |

This is done on <https://test.pypi.org> and <https://pypi.org> under the project's "Publishing" settings (or, for a first-ever publish of a new project name, under the account's pending-publisher settings). No agent or CI job can perform this step — it requires an authenticated human with access to the PyPI/TestPyPI account.

Optionally, add a required reviewer to the `pypi` GitHub environment (Settings → Environments → `pypi` → Required reviewers) for a manual approval gate before every production publish.

## Release checklist

1. **Update the version** in `pyproject.toml` (`[project] version = "X.Y.Z"`). This is the single source of truth — `coinfosim.__version__` reads it back via `importlib.metadata` once installed.
2. **Run the full test suite**: `python -m pytest` (includes the slow wheel-install and real-mode end-to-end tests; add `-m "not slow"` for a faster pass during iteration).
3. **Build**: `python -m build`.
4. **Check**: `python -m twine check dist/*`.
5. **TestPyPI prerelease**: tag a prerelease (`vX.Y.ZaN`, `vX.Y.ZbN`, or `vX.Y.ZrcN`) or run the `Publish to TestPyPI` workflow manually (`workflow_dispatch`).
6. **Install from TestPyPI in a clean environment** and smoke-test:
   ```bash
   python -m venv /tmp/testpypi-check
   /tmp/testpypi-check/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ coinfosim
   /tmp/testpypi-check/bin/coinfosim --version
   /tmp/testpypi-check/bin/coinfosim scenario run occupancy --mode smoke
   ```
7. **Tag the release** (`vX.Y.Z`) and **publish a GitHub Release** from that tag. This is what triggers `publish-pypi.yml`.
8. **PyPI Trusted Publishing** runs automatically (with an optional manual approval gate — see above); it builds once and publishes the exact artifacts it built and `twine check`-ed, never rebuilding after approval.
9. **Verify the PyPI project page** (`https://pypi.org/project/coinfosim/`) and `pip install coinfosim` in a fresh environment.
10. **Verify every dataset Pages URL** resolves (`https://paulorenatoaz.github.io/coinfosim/datasets/<dataset>/<file>` for all three datasets) and that `datasets/manifest.json` matches the packaged catalog.
11. **Publish/update the GitHub Pages home**:
    ```bash
    coinfosim publish pages --push
    ```

## What each workflow does

- **`ci.yml`** — unit and tiny-config tests across the OS/Python support matrix on every push/PR to `main`. Never runs `full`, `full-scale`, or `strict` scenario budgets.
- **`package.yml`** — builds sdist+wheel, `twine check`s them, installs the wheel into a clean environment, and runs CLI smoke tests, on every push/PR to `main`.
- **`publish-testpypi.yml`** — manual dispatch or a prerelease tag; builds once, publishes to TestPyPI via OIDC.
- **`publish-pypi.yml`** — triggered only by a published GitHub Release; builds once, publishes to PyPI via OIDC.
- **`publish-pages.yml`** — on push to `main`, regenerates and pushes the `gh-pages` branch content (reports, datasets, manifest, home page) via `coinfosim publish pages`.
- **`static.yml`** — deploys the current `gh-pages` branch content to the live GitHub Pages site whenever that branch changes.

## Post-release

- Delete any release-preparation branch.
- Confirm `main` is unaffected (release workflows never rewrite `main`).
- If a mistake is found after publishing, PyPI does not allow re-uploading the same version — bump the version and release again; do not attempt to delete/replace a published release.
