### Data
The full dataset for this project is over 40GB, so I haven't included it in the repo. It can be found at the link below for download from kaggle.com if you're interested. Remember to update main.py with the appropriate data folder.
https://www.kaggle.com/datasets/indominousx86/iot-23-dataset-for-multiclass-classification?resource=download

### Layout
IoT/
├── data/
│   └── raw/
│       └── *.csv             # Raw IoT-23 dataset chunks
├── notebooks/
│   └── main.py               # Entry point for development/testing
├── src/
│   ├── preprocessing/
│   │   └── preprocessing.py  # Preprocessor class with fit_transform, etc.
│   ├── data/
│   │   └── loader.py         # Chunked loader for large CSVs
|   |   └── splitter.py       # 
│   └── model/
│       └── model.py          # Placeholder for ML model logic (training, evaluation)
├── requirements.txt
└── README.md
