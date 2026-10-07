# data/ddata.csv

100,000 synthetic cardiac-monitoring records, 8 columns: the seven features
(`age`, `af_diagnosis`, `ecg_reading`, `ppg_reading`, `signal_quality`,
`af_context_30min`, `confidence_score`) and the binary label `af_detected`.
This is the exact file the paper's tables were produced from. See the
"Data" section of the top-level README and `tools/generate_data.py` for how
it was made. No real patient data was involved.
