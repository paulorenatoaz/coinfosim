# Dataset Distribution

CoInfoSim's three built-in scenarios are anchored to real, tracked datasets. This document explains how those files are distributed, cached, and verified.

## Where files live

- **Source of truth for hashes/URLs**: `src/coinfosim/resources/datasets.json`, packaged with `coinfosim` and loaded via `importlib.resources` (works identically from a source checkout or an installed wheel). See `coinfosim.datasets.catalog`.
- **Runtime download source**: `https://paulorenatoaz.github.io/coinfosim/datasets/`, the CoInfoSim GitHub Pages site. There is no automatic fallback to UCI, HBiostat, or any other upstream host.
- **Committed raw files**: `data/raw/<dataset>/` in this repository, hashed byte-for-byte identically to the packaged catalog and to the published Pages mirror. `.gitattributes` marks these files `-text` so no platform ever rewrites their line endings.
- **Local cache**: `platformdirs.user_data_dir("coinfosim") / "datasets" / <dataset>` (e.g. `~/.local/share/coinfosim/datasets/occupancy` on Linux).
- **Published manifest**: `https://paulorenatoaz.github.io/coinfosim/datasets/manifest.json`, generated from the same packaged catalog (`coinfosim.publish.datasets`) — schema version, generation timestamp, CoInfoSim version, source commit, and per-file provenance.

## Resolution order

See the [README's dataset section](../README.md#dataset-download-and-cache-behavior) for the exact six-step order (`--data-dir` → configured path → `COINFOSIM_DATA_DIR` → platform cache → automatic download → source-checkout fallback). A present-but-hash-invalid file is never used silently at any step; an explicitly named location (`--data-dir`, config, or `COINFOSIM_DATA_DIR`) that fails verification raises immediately rather than falling through.

## Verification

Every download is streamed to a temporary file in the destination directory, verified against the pinned SHA-256 and size, and only then installed atomically with `os.replace`. An existing invalid file is moved to a timestamped quarantine name before a fresh copy is installed (unless `--force`, which overwrites directly after the new download verifies). Partial downloads are always cleaned up on failure. See `coinfosim.datasets.integrity` and `coinfosim.datasets.download`.

## Provenance and licenses

### Occupancy Detection

- UCI dataset ID 357, DOI `10.24432/C5X01N`, creator Luis Candanedo.
- License: **CC BY 4.0** (Creative Commons Attribution 4.0 International).
- Files: `datatraining.txt` (training pool), `datatest.txt` + `datatest2.txt` (fixed test set).
- Source: <https://archive.ics.uci.edu/dataset/357/occupancy+detection>

### UCI Air Quality

- UCI dataset ID 360, DOI `10.24432/C59K5F`, creator Saverio De Vito.
- License: displayed by UCI as **CC BY 4.0**. The data are intended for research use; please cite the associated publication (De Vito et al., *Sensors and Actuators B: Chemical*, 2008).
- File: `AirQualityUCI.csv`, retained byte-for-byte from the original UCI distribution.
- Source: <https://archive.ics.uci.edu/dataset/360/air+quality>

### SUPPORT2

- DOI `10.3886/ICPSR02957.v2`, creator credited by UCI: Frank Harrell.
- **License: no explicit redistribution license identified.** It is a public research dataset; source acknowledgment is required, following the acknowledgment policy of the original HBiostat dataset site. **SUPPORT2 must never be labeled CC BY 4.0 or any other open license** — CoInfoSim's catalog, provenance READMEs, and published Pages home page all record this explicitly rather than inventing a license.
- File: `support2.csv`, retained byte-for-byte.
- Sources: <https://hbiostat.org/data/>, <https://archive.ics.uci.edu/dataset/880/support2>

Full per-file SHA-256 hashes and sizes are in `src/coinfosim/resources/datasets.json`, each `data/raw/<dataset>/README.md`, and the published `datasets/manifest.json` — all three are tested to agree (`tests/test_dataset_raw_files.py`, `tests/test_publish_datasets.py`).

## Offline / private-mirror use

```bash
coinfosim scenario run occupancy --data-dir /research/data/occupancy --no-download
coinfosim dataset fetch support2 --base-url https://internal-mirror.example.org/coinfosim/datasets --destination /tmp/support2
```

`--base-url` and `--destination` are explicit, advanced/testing-only overrides; the production default is always the pinned CoInfoSim Pages URLs. `--no-download` fails clearly (exit code 3) instead of silently proceeding if the dataset cannot be resolved offline.
