# IoT-23 Network Traffic Classification

A comprehensive machine learning pipeline for classifying IoT device network traffic using the IoT-23 dataset.

## Dataset Citation

If you are using this dataset for your research, please reference it as:

"Sebastian Garcia, Agustin Parmisano, & Maria Jose Erquiaga. (2020). IoT-23: A labeled dataset with malicious and benign IoT network traffic (Version 1.0.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4743746"

This dataset is created by converting the zeek network logs to csv files. To access the zeek logs and .pcap files refer to https://www.stratosphereips.org/datasets-iot23

## Overview

This project implements a memory conscious system for classifying network traffic as benign or malicious. It's designed to handle the full 40GB IoT-23 dataset through chunked processing and incremental learning.

## Key Features

- **Efficient Data Handling**: Processes large CSV files in manageable chunks
- **Advanced Feature Engineering**: Extracts network and time-based features
- **Multiple Classification Models**: Includes Random Forest, KNN, Logistic Regression, SVM, GMM
- **Incremental Learning**: Trains models in batches to handle large datasets
- **Model Evaluation and Analysis**: Generates performance metrics and visualizations

## Technical Components

### Data Processing

- **Chunked Loading**: Processes large files in configurable chunks
- **Parallel Processing**: Leverages multiple cores for faster data handling
- **Smart Caching**: Caches processed data to avoid redundant computation
- **Missing Value Handling**: Uses median for numerical and mode for categorical features

### Feature Engineering

- **Network Features**: Extracts traffic patterns like bytes/packets ratios and rates
- **Time Features**: Captures temporal patterns with cyclical time encodings
- **Protocol Encoding**: One-hot encodes protocol information

## Getting Started

### Prerequisites

- Python 3.10+
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd iot-classification

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Download the IoT-23 dataset from [Kaggle](https://www.kaggle.com/datasets/indominousx86/iot-23-dataset-for-multiclass-classification) and extract it to a directory of your choice. 

The original dataset is also available from [Zenodo](http://doi.org/10.5281/zenodo.4743746).

### Basic Usage

```bash
# Train models using default configuration
python main.py --data_dir path/to/data

# Use specific configuration file
python run.py --config config.yaml
```

### Configuration

The system can be configured through `config.yaml`:

```yaml
# Data settings
data:
  path: "path/to/data"
  file_pattern: "*.csv"
  num_files: 15  # Limit number of files to process
  chunk_size: 50000  # Rows per chunk
  max_rows: 500000  # Total rows to use
  balance_classes: true
  cache_processed: true

# Model training settings
training:
  output_dir: "models"
  seed: 42
  models:
    random_forest:
      enabled: true
      n_estimators: 100
      max_depth: 10
    svm:
      enabled: false  # Disable slow models
    # other model configurations...
```

## Advanced Usage

### Incremental Training

For very large datasets, use incremental training with batch processing:

```bash
python run.py --batch_size 3
```

This processes files in batches and updates models incrementally, making it possible to handle datasets larger than available memory.

### Model Evaluation

Evaluation metrics and visualizations are saved to the output directory:

- Confusion matrices
- ROC curves
- Feature importance plots
- Model comparison summaries

## Project Structure

```
IoT/
├── data/                    # Data storage (not included in repo)
├── notebooks/               # Analysis notebooks
├── src/
│   ├── data/                # Data loading and splitting
│   ├── features/            # Feature extraction
│   ├── models/              # Model implementation
│   └── preprocessing/       # Data preprocessing
├── utils/                   # Utility functions
├── config.yaml              # Configuration file
└── run.py                   # Entry point for incremental training
```

## Customization

### Adding New Features

Extend the `FeatureExtractor` base class:

```python
from src.features.base import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__("my_feature_extractor")
    
    def extract_features(self, df):
        # Add feature extraction logic
        return df_with_features
```

### Adding New Models

Extend the `BaseClassifier` class:

```python
from src.models.classifier import BaseClassifier

class MyModel(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = YourModelImplementation(**kwargs)
```

## Performance Considerations

- SVM training can be slow on large datasets (disable in config.yaml)
- Enable caching for faster retraining when experimenting