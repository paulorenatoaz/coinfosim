# SUPPORT2 raw data

`support2.csv` is the canonical ASCII distribution downloaded from the
Vanderbilt University Department of Biostatistics data repository:

- Dataset page: <https://hbiostat.org/data/>
- Archive: <https://hbiostat.org/data/repo/support2csv.zip>
- UCI mirror/description: <https://archive.ics.uci.edu/dataset/880/support2>
- DOI: <https://doi.org/10.3886/ICPSR02957.v2>
- Creator credited by UCI: Frank Harrell
- Acquired: 2026-07-14
- Size: 3141732 bytes
- SHA-256: `79621945edf2a5c8dc36359684ff356d3c6025e773ba4fefac26f865f7894c78`

**License / redistribution status:** no explicit redistribution license was
identified for SUPPORT2. It is a public research dataset; source
acknowledgment is required. Follow the acknowledgment policy of the original
HBiostat dataset site (<https://hbiostat.org/data/>) when using this file. Do
not describe this dataset as CC BY 4.0 or any other open license.

The file is retained byte-for-byte. `.gitattributes` marks it `-text` so it is
never rewritten by Git line-ending normalization on any platform; the SHA-256
above is computed from the exact committed bytes. It has 9,105 data rows. Its
header advertises 47 columns, but every data record contains 48 fields: the
leading field is an unnamed sequential patient identifier. CoInfoSim
reconstructs this field explicitly as `id`; it does not rely on pandas'
implicit index handling.

The scenario derives `death_180d` from `death` and `d.time` (`d.time` in the
source header). `hospdead` is an alternative in-hospital outcome and is neither
the primary target nor a predictor. See the dataset report for the complete
cohort, leakage, split, and preprocessing protocol.

## Distribution

This file is mirrored, byte-for-byte, on the CoInfoSim GitHub Pages site at:

```
https://paulorenatoaz.github.io/coinfosim/datasets/support2/support2.csv
```

`coinfosim scenario run support2` downloads and hash-verifies this file
automatically from that location; the pinned hash above is also recorded in
the installed package's dataset catalog
(`src/coinfosim/resources/datasets.json`).
