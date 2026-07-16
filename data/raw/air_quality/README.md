# UCI Air Quality source data

- **Dataset:** Air Quality, UCI Machine Learning Repository, dataset ID 360
- **DOI:** `10.24432/C59K5F`
- **Official dataset page:** https://archive.ics.uci.edu/dataset/360/air+quality
- **Official download archive:** https://archive.ics.uci.edu/static/public/360/air+quality.zip
- **Creator/donor credited by UCI:** Saverio Vito (the associated publication lists S. De Vito)
- **Associated publication:** S. De Vito, E. Massera, M. Piga, L. Martinotto,
  and G. Francia, “On field calibration of an electronic nose for benzene
  estimation in an urban pollution monitoring scenario,” *Sensors and
  Actuators B: Chemical*, 129(2), 2008.
- **Acquired:** 2026-07-14
- **Original filename:** `AirQualityUCI.csv`
- **Size:** 785065 bytes
- **Committed SHA-256:**
  `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`

`.gitattributes` marks this file `-text` so it is never rewritten by Git
line-ending normalization on any platform; the SHA-256 above is computed
from the exact committed bytes.

This file is mirrored, byte-for-byte, on the CoInfoSim GitHub Pages site at:

```
https://paulorenatoaz.github.io/coinfosim/datasets/air-quality/AirQualityUCI.csv
```

`coinfosim scenario run air-quality` downloads and hash-verifies this file
automatically from that location; the pinned hash above is also recorded in
the installed package's dataset catalog
(`src/coinfosim/resources/datasets.json`).

UCI identifies this dataset as CC BY 4.0 and requests appropriate attribution.
The UCI description also notes research-use provenance for these field sensor
measurements. This repository preserves the original CSV from UCI without
cleaning or replacement.

CoInfoSim uses this file to construct the dataset-anchored Air Quality
classification scenario. Exactly five PT08 metal-oxide sensor responses form
the classifier input. `C6H6(GT)` is retained only as the benzene reference used
to define the training-period experimental target; it is excluded from input
features.

The five input channels, in report and model order, are:

1. `PT08.S1(CO)`
2. `PT08.S2(NMHC)`
3. `PT08.S3(NOx)`
4. `PT08.S4(NO2)`
5. `PT08.S5(O3)`

After sentinel replacement, the protocol keeps complete cases for these five
channels plus `C6H6(GT)`. The first 80% of the resulting chronological cohort is
the training reservoir and the final 20% is the fixed future real test set. The
binary threshold is the training-only 75th percentile of `C6H6(GT)`; values
greater than or equal to the threshold receive the positive label. Channel
standardization uses training-only means and population standard deviations
(`ddof=0`). This experimental target is not a health or regulatory limit.
