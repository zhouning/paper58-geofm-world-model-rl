# Data and Code Availability

All source data used in this study are publicly available. AlphaEarth embedding fields were obtained from Google Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, CC-BY-4.0). ESRI product land-cover labels were obtained from the ESRI Global LULC 10 m Time Series dataset (`projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`) and mapped to the six-class taxonomy described in the manuscript. NASA HLS inputs used for the Prithvi diagnostic were accessed through Google Earth Engine.

The public repository is available at https://github.com/zhouning/paper58-geofm-world-model-rl. It contains the processed embedding caches, trained GeoFM-LDN checkpoints, result CSV/JSON files, figure source data, manuscript figures, and scripts needed to regenerate the tables and diagnostics reported in the article. The repository includes analysis scripts for paired inference, ESRI product-label change validation, GeoSOS-FLUS per-area comparison, cosine-to-accuracy diagnostics, multi-step rollout diagnostics, per-year decoder evaluation, manuscript consistency checks, and figure/table regeneration.

No human-participant, animal-subject, or access-restricted data are used.

## Recommended before final upload

Create a Zenodo or institutional repository DOI snapshot for the exact submitted GitHub state before final upload, then add the DOI to the submission metadata if available. The current manuscript uses the GitHub URL because the DOI does not exist at package-preparation time.