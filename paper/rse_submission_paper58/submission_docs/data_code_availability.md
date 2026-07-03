# Data and Code Availability

All source data used in this study are publicly available. AlphaEarth embedding fields were obtained from Google Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, CC-BY-4.0). Reference land-cover labels were obtained from the ESRI Global LULC 10 m Time Series dataset (`projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`) and mapped to the six-class taxonomy described in the manuscript. NASA HLS inputs used for the Prithvi diagnostic were accessed through Google Earth Engine.

The public repository is available at:

`https://github.com/zhouning/paper58-geofm-world-model-rl`

The repository contains the processed embedding caches, trained GeoFM-LDN checkpoints, result CSV/JSON files, figure source data, manuscript figures, and scripts needed to regenerate the tables and diagnostics reported in the article. It includes analysis scripts for paired inference, independent change validation, GeoSOS-FLUS per-area comparison, cosine-to-accuracy diagnostics, multi-step rollout diagnostics, per-year decoder evaluation, manuscript consistency checks, and figure/table regeneration. It also contains cache-aligned retraining outputs and trained dynamics checkpoints. These materials are distributed in the repository's documented directory structure so that the tables and diagnostics reported in the article can be regenerated.

No human-participant, animal-subject, or access-restricted data are used. A DOI-archived snapshot of the public repository will be added if required by the journal or before final publication.
