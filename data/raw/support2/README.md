# SUPPORT2 raw data

`support2.csv` is the canonical ASCII distribution downloaded from the
Vanderbilt University Department of Biostatistics data repository:

- Dataset page: <https://hbiostat.org/data/>
- Archive: <https://hbiostat.org/data/repo/support2csv.zip>
- UCI mirror/description: <https://archive.ics.uci.edu/dataset/880/support2>
- DOI: <https://doi.org/10.3886/ICPSR02957.v2>
- SHA-256: `79621945edf2a5c8dc36359684ff356d3c6025e773ba4fefac26f865f7894c78`

The file is retained byte-for-byte. It has 9,105 data rows. Its header
advertises 47 columns, but every data record contains 48 fields: the leading
field is an unnamed sequential patient identifier. CoInfoSim reconstructs this
field explicitly as `id`; it does not rely on pandas' implicit index handling.

The scenario derives `death_180d` from `death` and `d.time` (`d.time` in the
source header). `hospdead` is an alternative in-hospital outcome and is neither
the primary target nor a predictor. See the dataset report for the complete
cohort, leakage, split, and preprocessing protocol.
