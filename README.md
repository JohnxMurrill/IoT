project/
│
├── data/               # raw and processed data
├── notebooks/          # exploratory analysis
├── src/
│   ├── data_loader.py  # memory-aware CSV parsing
│   ├── preprocess.py   # feature engineering
│   ├── model.py        # training & evaluation
│   └── predict.py      # batch or live predictions
├── models/             # serialized models
├── results/            # saved evaluation metrics
└── run_pipeline.py     # entry point script