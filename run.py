#!/usr/bin/env python
"""
Script to run the IoT-23 training pipeline with incremental learning to handle large datasets.
"""

import os
import sys
from pathlib import Path
import logging
import time
import argparse
import yaml
import numpy as np
import joblib
from sklearn.base import clone

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir  # Assuming run.py is in the project root
sys.path.append(str(project_root))

# Import project components
try:
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

def save_models(models, output_dir):
    """Save trained models to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        model_path = output_dir / f"{name}.joblib"
        try:
            joblib.dump(model, model_path)
            logger.info(f"Model {name} saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {name}: {e}")

def load_models(output_dir, model_configs=None):
    """Load trained models from disk, or initialize new ones if not found."""
    output_dir = Path(output_dir)
    models = {}
    
    # Define model classes
    model_classes = {
        'random_forest': RandomForestModel,
        'svm': SVMModel,
        'knn': KNNModel,
        'gaussian_nb': GaussianNBModel,
        'logistic_regression': LogisticRegressionModel,
        'gmm': GMMClassifier
    }
    
    # Check if models exist
    for model_name, model_class in model_classes.items():
        model_path = output_dir / f"{model_name}.joblib"
        
        if model_path.exists():
            try:
                # Load existing model
                logger.info(f"Loading existing model: {model_name}")
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                
                # If loading fails, initialize a new model
                if model_configs and model_name in model_configs:
                    config = model_configs[model_name].copy()
                    
                    # Remove 'enabled' flag if present
                    if 'enabled' in config:
                        del config['enabled']
                        
                    # Initialize a new model with config
                    logger.info(f"Initializing new {model_name} model after failed load")
                    models[model_name] = model_class(**config)
        elif model_configs and model_name in model_configs:
            # Initialize a new model if configs are provided
            config = model_configs[model_name].copy()
            
            # Skip disabled models
            if not config.get('enabled', True):
                logger.info(f"Skipping disabled model: {model_name}")
                continue
                
            # Remove 'enabled' flag if present
            if 'enabled' in config:
                del config['enabled']
                
            # Filter parameters for models that don't support random_state
            if model_name in ['knn', 'gaussian_nb', 'gmm'] and 'random_state' in config:
                del config['random_state']
                
            # Initialize a new model
            logger.info(f"Initializing new model: {model_name}")
            
            # Special handling for GMM with n_components
            if model_name == 'gmm' and 'n_components' in config:
                # We'll set the actual n_components when we have the data
                models[model_name] = model_class(**config)
            else:
                models[model_name] = model_class(**config)
    
    return models

def incremental_train_iot23_models(config=None):
    """
    Train models on IoT-23 dataset using incremental learning to handle large datasets.
    
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
    
    # Incremental learning settings
    batch_size = data_config.get('batch_size', 3)  # Number of files to process per batch
    
    start_time = time.time()
    logger.info("Starting IoT-23 incremental training pipeline")
    
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
    
    # Initialize feature extractors
    enabled_extractors = features_config.get('extractors', ['network_traffic', 'time_based'])
    extractors = []
    
    if 'network_traffic' in enabled_extractors:
        extractors.append(NetworkTrafficFeatures())
        
    if 'time_based' in enabled_extractors:
        extractors.append(TimeBasedFeatures())
    
    # Create composite extractor
    composite_extractor = CompositeFeatureExtractor(extractors)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Create model configs with default values
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
            'enabled': True,
            'kernel': 'rbf',
            'C': 1.0
        },
        'gaussian_nb': {
            'enabled': True
        },
        'gmm': {
            'enabled': True,
            'n_components': 5
        }
    }
    
    # Override with user config
    for model_name, default_config in default_models.items():
        user_config = model_configs.get(model_name, {})
        default_models[model_name].update(user_config)
    
    # Add random state to models that support it
    for model_name in ['random_forest', 'logistic_regression', 'svm']:
        if 'random_state' not in default_models[model_name]:
            default_models[model_name]['random_state'] = seed
    
    # Process files in batches
    total_batches = (len(files) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(files))
        batch_files = files[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch_files)} files")
        
        # Get data directory
        data_dir = batch_files[0].parent
        
        # Initialize data loader
        loader = DataLoader(
            data_dir,
            chunk_size=chunk_size,
            n_jobs=-1,
            cache_processed=cache_processed
        )
        
        # Override file paths with our batch selection
        loader.file_paths = batch_files
        
        # Process batch files
        logger.info(f"Processing files in batch {batch_idx+1}...")
        processed_data = loader.process_files_parallel(
            batch_files,
            feature_extractor=composite_extractor,
            max_rows_total=max_rows,
            balance_classes=balance_classes
        )
        
        if processed_data.empty:
            logger.warning(f"No data was processed in batch {batch_idx+1}. Skipping batch.")
            continue
            
        logger.info(f"Batch {batch_idx+1} processed data shape: {processed_data.shape}")
        
        # Print class distribution
        if target_col in processed_data.columns:
            class_dist = processed_data[target_col].value_counts()
            logger.info(f"Batch {batch_idx+1} class distribution:\n{class_dist}")
        
        # Split data - check if time-based split is possible
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
        
        logger.info(f"Batch {batch_idx+1} split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Preprocess data
        logger.info(f"Preprocessing data for batch {batch_idx+1}...")
        try:
            # Fit and transform training data
            X_train, y_train = preprocessor.fit_transform(
                train_df, 
                target_col, 
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                drop_na=False  # Don't drop rows with NA values
            )
            
            # Transform validation and test data
            X_val, y_val = preprocessor.transform(val_df)
            X_test, y_test = preprocessor.transform(test_df)
            
            logger.info(f"Batch {batch_idx+1} training data shapes - X: {X_train.shape}, y: {y_train.shape}")
            logger.info(f"Batch {batch_idx+1} validation data shapes - X: {X_val.shape}, y: {y_val.shape}")
            logger.info(f"Batch {batch_idx+1} test data shapes - X: {X_test.shape}, y: {y_test.shape}")
            
            # Ensure output directory exists
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing models or initialize new ones
            models = load_models(output_dir_path, default_models)
            
            # Special handling for GMM n_components
            if 'gmm' in models:
                # Ensure n_components doesn't exceed unique classes
                gmm_model = models['gmm']
                if hasattr(gmm_model, 'n_components'):
                    gmm_model.n_components = min(gmm_model.n_components, len(np.unique(y_train)))
            
            # Train models on this batch
            logger.info(f"Training models on batch {batch_idx+1}...")
            for name, model in models.items():
                try:
                    logger.info(f"Training {name} on batch {batch_idx+1}...")
                    
                    # For first batch or models that don't support partial_fit, use fit
                    if batch_idx == 0 or not hasattr(model, 'partial_fit'):
                        model.fit(X_train, y_train)
                    else:
                        # For models with partial_fit, use that for incremental learning
                        # Note: Not all sklearn models support partial_fit
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_train, y_train)
                        else:
                            # For models without partial_fit, we need to retrain on all data
                            # This is a limitation for some model types
                            model.fit(X_train, y_train)
                    
                    logger.info(f"Finished training {name} on batch {batch_idx+1}")
                except Exception as e:
                    logger.error(f"Error training {name} on batch {batch_idx+1}: {e}")
            
            # Save models after each batch
            save_models(models, output_dir_path)
            
            # Evaluate models on test data from this batch
            logger.info(f"Evaluating models on batch {batch_idx+1} test data...")
            for name, model in models.items():
                try:
                    if hasattr(model, 'evaluate'):
                        eval_result = model.evaluate(X_test, y_test)
                        logger.info(f"Model {name} batch {batch_idx+1} accuracy: {eval_result['accuracy']:.4f}")
                    else:
                        # Fall back to manual evaluation
                        y_pred = model.predict(X_test)
                        accuracy = np.mean(y_test == y_pred)
                        logger.info(f"Model {name} batch {batch_idx+1} accuracy: {accuracy:.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating {name} on batch {batch_idx+1}: {e}")
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx+1}: {e}")
            continue
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Incremental training pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Models saved to {output_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Train models on IoT-23 dataset using incremental learning')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Number of files to process per batch (overrides config)')
    
    args = parser.parse_args()
    
    if not MODULE_IMPORT_SUCCESS:
        print("Failed to import required modules. Please check your project structure.")
        return
        
    # Load configuration
    config = load_config(args.config)
    
    # Override batch size if specified
    if args.batch_size is not None:
        if 'data' not in config:
            config['data'] = {}
        config['data']['batch_size'] = args.batch_size
    
    # Train models incrementally
    success = incremental_train_iot23_models(config)
    
    if success:
        print("Incremental training completed successfully! Check the logs for details.")
    else:
        print("Training failed. Check the logs for error details.")

if __name__ == "__main__":
    main()