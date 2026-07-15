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
- **Committed SHA-256:**
  `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`

UCI identifies this dataset as CC BY 4.0 and requests appropriate attribution.
The UCI description also notes research-use provenance for these field sensor
measurements. This repository preserves the original CSV from UCI without
cleaning or replacement.

CoInfoSim uses this file to construct the dataset-anchored Air Quality
classification scenario. Exactly five PT08 metal-oxide sensor responses form
the classifier input. `C6H6(GT)` is retained only as the benzene reference used
to define the training-period experimental target; it is excluded from input
features.
