# UCI Occupancy Detection source data

- **Dataset:** Occupancy Detection, UCI Machine Learning Repository, dataset ID 357
- **DOI:** `10.24432/C5X01N`
- **Official dataset page:** https://archive.ics.uci.edu/dataset/357/occupancy+detection
- **Creator/donor credited by UCI:** Luis Candanedo
- **License:** Creative Commons Attribution 4.0 International (`CC BY 4.0`),
  https://creativecommons.org/licenses/by/4.0/. Attribution required.
- **Acquired:** 2026-07-14

## Files

| Filename | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `datatraining.txt` | 596674 | `b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56` |
| `datatest.txt` | 200766 | `1b92c7c1b2838963464fa891a610cf3c5db4becb7189189b29b330107a584c7f` |
| `datatest2.txt` | 699664 | `d026d1bd5aeccd4aff4f3b3710d48e40613bd5fc370db7e61bbdcaa50d985095` |

This repository preserves the original files from UCI without cleaning or
replacement. `.gitattributes` marks these files `-text` so they are never
rewritten by Git line-ending normalization on any platform; the SHA-256
values above are computed from the exact committed bytes.

## Original protocol

- `datatraining.txt` is the finite training pool.
- `datatest.txt` and `datatest2.txt` together form the fixed test set.

Standardization parameters (per-channel mean and standard deviation) are
estimated from the training pool only and then applied to both splits. See
`coinfosim.datasets.occupancy` and the generated dataset report for the
complete loading and standardization protocol.

## Distribution

These files are mirrored, byte-for-byte, on the CoInfoSim GitHub Pages site
at:

```
https://paulorenatoaz.github.io/coinfosim/datasets/occupancy/
```

`coinfosim scenario run occupancy` downloads and hash-verifies these files
automatically from that location; the pinned hashes above are also recorded
in the installed package's dataset catalog
(`src/coinfosim/resources/datasets.json`).
