#!/usr/bin/env python
"""
Script to run the IoT-23 training pipeline with configuration from YAML file.
"""

import os
import sys
from pathlib import Path
import logging
import time
import argparse
import yaml
import numpy as np
import pandas as pd

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir  # Assuming run.py is in the project root
sys.path.append(str(project_root))

# Import project components
try:
    # Try to import required modules
    from src.data.loader import DataLoader
    from src.features.composite import CompositeFeatureExtractor
    from src.features.network import NetworkTrafficFeatures
    from src.features.time import TimeBasedFeatures
    from src.preprocessing.preprocessing import DataPreprocessor
    from src.data.splitter import DataSplitter
    from src.models.classifier import (
        RandomForestModel, 
        SVMModel, 
        GaussianNBModel,
        LogisticRegressionModel,
        KNNModel, 
        GMMClassifier
    )
    from src.models.training import ModelTrainer
    MODULE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    MODULE_IMPORT_SUCCESS = False
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("iot23_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('iot23_pipeline')

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found. Using default settings.")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def find_data_files(data_path=None, pattern="*.csv"):
    """Find IoT-23 dataset files to process."""
    
    # Start with current directory if none specified
    if data_path is None:
        data_path = Path.cwd()
    else:
        data_path = Path(data_path)
        
    # If data_path is a file, return it directly
    if data_path.is_file() and data_path.suffix.lower() == '.csv':
        return [data_path]
        
    # Check for CSV files in the directory
    files = list(data_path.glob(pattern))
    
    if not files:
        # Check common subdirectories
        potential_dirs = [
            data_path / "data",
            data_path / "IoT-23",
            data_path / "IoT-23-Max",
            data_path / "data" / "data" / "IoT-23-Max"
        ]
        
        for directory in potential_dirs:
            if directory.exists():
                files = list(directory.glob(pattern))
                if files:
                    logger.info(f"Found files in {directory}")
                    return files
                    
    return files

def analyze_and_fix_labels(df, target_col='label'):
    """
    Analyze label distribution and fix any inconsistencies.
    
    Args:
        df: DataFrame to analyze
        target_col: Target column name
    
    Returns:
        DataFrame with fixed labels
    """
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found in DataFrame")
        return df
    
    logger.info("Analyzing label distribution...")
    
    # Make a copy to avoid modifying the original
    df_fixed = df.copy()
    
    # Get original label distribution
    orig_dist = df_fixed[target_col].value_counts()
    logger.info(f"Original label distribution:\n{orig_dist}")
    
    # Check for case inconsistencies
    if pd.api.types.is_object_dtype(df_fixed[target_col]):
        # Get lowercase version
        df_fixed[target_col + '_lower'] = df_fixed[target_col].str.lower()
        
        # Compare original vs lowercase distributions
        lower_dist = df_fixed[target_col + '_lower'].value_counts()
        
        if len(orig_dist) != len(lower_dist):
            logger.warning(f"Found case inconsistencies in labels!")
            logger.info(f"After lowercase conversion, unique label count reduced from {len(orig_dist)} to {len(lower_dist)}")
            
            # Create a mapping of lowercase -> original case variations
            case_variations = {}
            for label in orig_dist.index:
                if isinstance(label, str):
                    lower = label.lower()
                    if lower not in case_variations:
                        case_variations[lower] = [label]
                    else:
                        case_variations[lower].append(label)
            
            # Log variations found
            variations_found = {k: v for k, v in case_variations.items() if len(v) > 1}
            if variations_found:
                logger.info(f"Case variations found: {variations_found}")
            
            # Normalize labels to lowercase
            df_fixed[target_col] = df_fixed[target_col + '_lower']
        
        # Drop the temporary column
        df_fixed = df_fixed.drop(columns=[target_col + '_lower'])
        
        # Apply standard label mapping (e.g., "normal" -> "benign")
        label_mapping = {
            'malicious': 'malicious',
            'benign': 'benign', 
            'normal': 'benign',
            'anomaly': 'malicious',
            'attack': 'malicious'
        }
        
        # Apply the mapping
        df_fixed[target_col] = df_fixed[target_col].map(
            lambda x: label_mapping.get(x, x) if isinstance(x, str) and x in label_mapping else x
        )
        
        # Log final distribution
        final_dist = df_fixed[target_col].value_counts()
        logger.info(f"Final label distribution after normalization:\n{final_dist}")
        
        # Check for highly imbalanced classes
        if len(final_dist) > 1:
            imbalance_ratio = final_dist.max() / final_dist.min()
            logger.info(f"Class imbalance ratio (majority:minority): {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 10:
                logger.warning("Severe class imbalance detected. Consider using balance_classes=True")
    
    return df_fixed

# Usage in run.py:
# After loading the data:
# processed_data = analyze_and_fix_labels(processed_data, target_col)

def train_iot23_models(config=None):
    """
    Train models on IoT-23 dataset using configuration.
    
    Args:
        config: Configuration dictionary (if None, defaults will be used)
    """
    # Use default config if none provided
    if config is None:
        config = {}
        
    # Extract settings from config with defaults
    data_config = config.get('data', {})
    features_config = config.get('features', {})
    training_config = config.get('training', {})
    
    # Data settings
    data_path = data_config.get('path', None)
    file_pattern = data_config.get('file_pattern', '*.csv')
    num_files = data_config.get('num_files', None)
    chunk_size = data_config.get('chunk_size', 50000)
    max_rows = data_config.get('max_rows', None)
    balance_classes = data_config.get('balance_classes', True)
    cache_processed = data_config.get('cache_processed', True)
    
    # Feature settings
    categorical_columns = features_config.get('categorical_columns', None)
    numerical_columns = features_config.get('numerical_columns', None)
    target_col = features_config.get('target_column', 'label')
    
    # Training settings
    output_dir = training_config.get('output_dir', 'models')
    seed = training_config.get('seed', 42)
    hyperparameter_tuning = training_config.get('hyperparameter_tuning', True)
    model_configs = training_config.get('models', {})
    
    start_time = time.time()
    logger.info("Starting IoT-23 training pipeline")
    
    # Find data files
    files = find_data_files(data_path, file_pattern)
    
    if not files:
        logger.error("No data files found. Please specify a valid path.")
        return False
        
    logger.info(f"Found {len(files)} data files")
    
    # Limit number of files if specified
    if num_files is not None and num_files < len(files):
        files = files[:num_files]
        logger.info(f"Using {num_files} files")
        
    # Get data directory
    data_dir = files[0].parent
    logger.info(f"Using data directory: {data_dir}")
    
    # Initialize data loader
    loader = DataLoader(
        data_dir,
        chunk_size=chunk_size,
        n_jobs=-1,
        cache_processed=cache_processed
    )
    
    # Override file paths with our selection
    loader.file_paths = files
    
    # Create feature extractors
    enabled_extractors = features_config.get('extractors', ['network_traffic', 'time_based'])
    extractors = []
    
    if 'network_traffic' in enabled_extractors:
        extractors.append(NetworkTrafficFeatures())
        
    if 'time_based' in enabled_extractors:
        extractors.append(TimeBasedFeatures())
    
    # Create composite extractor
    composite_extractor = CompositeFeatureExtractor(extractors)
    
    # Process files
    logger.info("Processing files...")
    processed_data = loader.process_files_parallel(
        files,
        feature_extractor=composite_extractor,
        max_rows_total=max_rows,
        balance_classes=balance_classes
    )
    processed_data = analyze_and_fix_labels(processed_data, target_col)
    
    if processed_data.empty:
        logger.error("No data was processed successfully.")
        return False
        
    logger.info(f"Combined processed data shape: {processed_data.shape}")
    
    # Print class distribution
    if target_col in processed_data.columns:
        class_dist = processed_data[target_col].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")
    else:
        logger.warning(f"Target column '{target_col}' not found in data. Available columns: {processed_data.columns.tolist()}")
        # Try to find an alternative target column
        potential_targets = ['label', 'detailed-label', 'class']
        for potential in potential_targets:
            if potential in processed_data.columns:
                target_col = potential
                logger.info(f"Using alternative target column: {target_col}")
                break
                
        if target_col not in processed_data.columns:
            logger.error("No suitable target column found in data.")
            return False
    
    # If columns not specified in config, auto-detect
    if categorical_columns is None:
        potential_categorical = [
            'proto', 'service', 'conn_state', 'local_orig', 'local_resp',
            'history', 'tunnel_parents'
        ]
        categorical_columns = [col for col in potential_categorical if col in processed_data.columns]
    else:
        # Ensure all specified columns exist
        categorical_columns = [col for col in categorical_columns if col in processed_data.columns]
    
    if numerical_columns is None:
        potential_numerical = [
            'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
            'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
            'total_bytes', 'bytes_ratio', 'total_pkts', 'pkts_ratio',
            'bytes_per_sec', 'pkts_per_sec', 'hour', 'day_of_week', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        numerical_columns = [col for col in potential_numerical if col in processed_data.columns]
    else:
        # Ensure all specified columns exist
        numerical_columns = [col for col in numerical_columns if col in processed_data.columns]
        
    logger.info(f"Using target column: {target_col}")
    logger.info(f"Using categorical columns: {categorical_columns}")
    logger.info(f"Using numerical columns: {numerical_columns}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Split data
    logger.info("Splitting data...")
    if 'timestamp' in processed_data.columns:
        logger.info("Performing time-based split")
        train_df, val_df, test_df = DataSplitter.time_based_split(
            processed_data, 'timestamp'
        )
    else:
        logger.info("Performing stratified split")
        train_df, val_df, test_df = DataSplitter.stratified_split(
            processed_data, target_col
        )
    
    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    try:
        # Fit and transform training data
        X_train, y_train = preprocessor.fit_transform(
            train_df, 
            target_col, 
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns
        )
        
        # Transform validation and test data
        X_val, y_val = preprocessor.transform(val_df)
        X_test, y_test = preprocessor.transform(test_df)
        
        logger.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        logger.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")
        logger.info(f"Test data shapes - X: {X_test.shape}, y: {y_test.shape}")
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model instances based on config
        logger.info("Initializing models...")
        models = {}
        
        # Default model configs
        default_models = {
            'random_forest': {
                'enabled': True,
                'n_estimators': 100,
                'max_depth': 10
            },
            'logistic_regression': {
                'enabled': True,
                'max_iter': 1000
            },
            'knn': {
                'enabled': True,
                'n_neighbors': 5
            },
            'svm': {
                'enabled': True if X_train.shape[0] < 100000 else False,
                'kernel': 'rbf',
                'C': 1.0,
                'probability': True
            },
            'gaussian_nb': {
                'enabled': True if X_train.shape[0] < 100000 else False
            },
            'gmm': {
                'enabled': True if X_train.shape[0] < 100000 else False,
                'n_components': 5
            }
        }
        
        # Override with user config
        for model_name, default_config in default_models.items():
            # Get config from user config if available
            user_config = model_configs.get(model_name, {})
            
            # Combine default with user config
            model_config = {**default_config, **user_config}
            
            # Skip disabled models
            if not model_config.get('enabled', True):
                logger.info(f"Skipping disabled model: {model_name}")
                continue
                
            # Remove 'enabled' flag before passing to model constructor
            if 'enabled' in model_config:
                del model_config['enabled']
                
            if 'random_state' not in model_config:
                # List of models that support random_state
                models_with_random_state = ['random_forest', 'logistic_regression', 'svm']
                
                # Only add random_state to models that support it
                if model_name in models_with_random_state:
                    model_config['random_state'] = seed
                    logger.debug(f"Added random_state={seed} to {model_name}")
                else:
                    logger.debug(f"Model {model_name} does not support random_state, skipping")
                
            # Create the model
            if model_name == 'random_forest':
                models[model_name] = RandomForestModel(**model_config)
            elif model_name == 'logistic_regression':
                models[model_name] = LogisticRegressionModel(**model_config)
            elif model_name == 'knn':
                models[model_name] = KNNModel(**model_config)
            elif model_name == 'svm':
                models[model_name] = SVMModel(**model_config)
            elif model_name == 'gaussian_nb':
                models[model_name] = GaussianNBModel(**model_config)
            elif model_name == 'gmm':
                # Adjust n_components to not exceed unique classes
                n_components = min(model_config.get('n_components', 5), len(np.unique(y_train)))
                model_config['n_components'] = n_components
                models[model_name] = GMMClassifier(**model_config)
                
        # Initialize model trainer
        trainer = ModelTrainer(models, output_dir=output_dir)
        
        # Train all models
        logger.info("Training models...")
        trainer.train_all(X_train, y_train)
        
        # Optional: Hyperparameter tuning
        if hyperparameter_tuning and 'random_forest' in models and X_train.shape[0] < 50000:
            logger.info("Tuning hyperparameters for Random Forest...")
            rf_param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
            trainer.tune_hyperparameters('random_forest', rf_param_grid, X_train, y_train)
        
        # Evaluate all models
        logger.info("Evaluating models...")
        evaluation_results = trainer.evaluate_all(X_test, y_test)
        
        # Print summary of results
        summary = trainer.summarize_results()
        if summary is not None:
            logger.info("Model performance summary:")
            logger.info("\n" + str(summary))
            
        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"Training pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Models and results saved to {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing or model training: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description='Train models on IoT-23 dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    if not MODULE_IMPORT_SUCCESS:
        print("Failed to import required modules. Please check your project structure.")
        return
        
    # Load configuration
    config = load_config(args.config)
    
    # Train models
    success = train_iot23_models(config)
    
    if success:
        print("Training completed successfully! Check the logs for details.")
    else:
        print("Training failed. Check the logs for error details.")

if __name__ == "__main__":
    main()