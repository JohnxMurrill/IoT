import sys
import logging
import argparse
from pathlib import Path
import numpy as np

# Ensure the project root is in the path
current_dir = Path(__file__).parent
root_dir = current_dir.parent  # Adjust if needed
sys.path.append(str(root_dir))

from src.data.loader import DataLoader
from src.data.splitter import DataSplitter
from src.features.composite import CompositeFeatureExtractor
from src.features.network import NetworkTrafficFeatures 
from src.features.time import TimeBasedFeatures
from src.preprocessing.preprocessing import DataPreprocessor
from src.models.classifier import (
    RandomForestModel, 
    SVMModel, 
    KNNModel, 
    GaussianNBModel,
    LogisticRegressionModel,
    GMMClassifier
)
from src.models.training import ModelTrainer
import os

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='IoT-23 Classification Pipeline')
    parser.add_argument('--data_dir', type=str, default=None, 
                        help='Directory containing IoT-23 dataset files')
    parser.add_argument('--file_pattern', type=str, default="*.csv", 
                        help='Pattern to match dataset files (default: *.csv)')
    parser.add_argument('--single_file', type=str, default=None,
                        help='Process a single specific file instead of a directory')
    parser.add_argument('--num_files', type=int, default=None, 
                        help='Number of files to process (default: all available)')
    parser.add_argument('--chunk_size', type=int, default=50000, 
                        help='Chunk size for processing (default: 50,000)')
    parser.add_argument('--max_rows', type=int, default=None, 
                        help='Maximum rows to use in total (default: None)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--balance_classes', action='store_true',
                        help='Balance classes in the dataset')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching of processed data')
    return parser.parse_args()

def main():
    # Parse arguments or use defaults
    try:
        args = parse_arguments()
    except:
        # If running in an environment that doesn't support argparse (like a notebook)
        class Args:
            pass
        args = Args()
        args.data_dir = None
        args.file_pattern = "*.csv"
        args.single_file = None
        args.num_files = None
        args.chunk_size = 50000
        args.max_rows = None
        args.output_dir = 'models'
        args.seed = 42
        args.balance_classes = True
        args.no_cache = False
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('iot23_framework')
    
    # Determine data directory
    if args.single_file:
        data_path = Path(args.single_file)
        data_dir = data_path.parent
    elif args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Look for data in common locations
        potential_locations = [
            Path("./data"),
            Path("./IoT/data"),
            Path("./data/data/IoT-23-Max"),
            Path(current_dir) / "data",
            current_dir  # Look in current directory
        ]
        
        data_dir = None
        for loc in potential_locations:
            if loc.exists():
                # Check if we have CSV files here
                if list(loc.glob(args.file_pattern)):
                    data_dir = loc
                    break
                    
        if data_dir is None:
            # Use current directory as fallback
            data_dir = current_dir
            logger.warning(f"No data directory specified, using current directory: {data_dir}")
    
    logger.info(f"Using data directory: {data_dir}")
    
    # Initialize data loader
    loader = DataLoader(
        data_dir, 
        chunk_size=args.chunk_size,
        cache_processed=not args.no_cache
    )
    
    # If single file specified, filter to just that file
    if args.single_file:
        single_file_path = Path(args.single_file)
        if single_file_path.exists():
            loader.file_paths = [single_file_path]
        else:
            logger.error(f"Specified file not found: {args.single_file}")
            return
    
    # Get files info
    files_info = loader.get_files_info()
    if files_info.empty:
        logger.error(f"No files matching '{args.file_pattern}' found in {data_dir}")
        return
        
    logger.info(f"Found {len(files_info)} files:")
    for i, row in files_info.iterrows():
        logger.info(f"  {row['filename']} - {row['size_mb']:.2f} MB")
    
    # Create feature extractors
    network_features = NetworkTrafficFeatures()
    time_features = TimeBasedFeatures()
    
    # Create composite extractor
    composite_extractor = CompositeFeatureExtractor([
        network_features,
        time_features
    ])
    
    # Process files
    if len(loader.file_paths) > 0:
        # Limit number of files if specified
        if args.num_files is not None and args.num_files < len(loader.file_paths):
            selected_files = loader.file_paths[:args.num_files]
        else:
            selected_files = loader.file_paths
        
        logger.info(f"Will process {len(selected_files)} files")
        
        # Process files in parallel
        processed_data = loader.process_files_parallel(
            selected_files,
            feature_extractor=composite_extractor,
            max_rows_total=args.max_rows,
            balance_classes=args.balance_classes
        )
        
        if processed_data.empty:
            logger.error("No data was processed successfully. Check file format or paths.")
            return
            
        logger.info(f"Combined processed data shape: {processed_data.shape}")
        
        # Check class distribution
        if 'label' in processed_data.columns:
            class_dist = processed_data['label'].value_counts()
            logger.info(f"Class distribution:\n{class_dist}")
        
        # Define feature columns based on available columns
        all_columns = processed_data.columns.tolist()
        
        # List potential categorical and numerical columns
        potential_categorical = [
            'proto', 'service', 'conn_state', 'local_orig', 'local_resp',
            'history', 'tunnel_parents'
        ]
        
        potential_numerical = [
            'id.orig_p', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
            'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
            'total_bytes', 'bytes_ratio', 'total_pkts', 'pkts_ratio',
            'bytes_per_sec', 'pkts_per_sec', 'hour', 'day_of_week', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Filter to only include columns that exist in the processed data
        categorical_columns = [col for col in potential_categorical if col in all_columns]
        numerical_columns = [col for col in potential_numerical if col in all_columns]
        
        # Target column (adjust if needed)
        target_col = 'label' if 'label' in all_columns else 'detailed-label'
        
        if target_col not in all_columns:
            logger.error(f"Target column '{target_col}' not found in data. Available columns: {all_columns}")
            return
            
        logger.info(f"Using target column: {target_col}")
        logger.info(f"Using categorical columns: {categorical_columns}")
        logger.info(f"Using numerical columns: {numerical_columns}")
        
        # Preprocess for ML
        preprocessor = DataPreprocessor()
        
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
        
        logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Prepare data for model training
        try:
            # Fit and transform on training data
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
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create model instances
            models = {
                'random_forest': RandomForestModel(n_estimators=100, max_depth=10, random_state=args.seed),
                'logistic_regression': LogisticRegressionModel(max_iter=1000, random_state=args.seed),
                'knn': KNNModel(n_neighbors=5)
            }
            
            # Add more complex models only if dataset is not too large
            if X_train.shape[0] < 100000:  # Arbitrary threshold
                models.update({
                    'svm': SVMModel(kernel='rbf', C=1.0, probability=True, random_state=args.seed),
                    'gaussian_nb': GaussianNBModel(),
                    'gmm': GMMClassifier(n_components=min(5, len(np.unique(y_train))))
                })
            
            # Initialize model trainer
            trainer = ModelTrainer(models, output_dir=output_dir)
            
            # Train all models
            logger.info("Training models...")
            trainer.train_all(X_train, y_train)
            
            # Optional: Hyperparameter tuning for Random Forest
            # Only if dataset is not too large
            if X_train.shape[0] < 50000:  # Arbitrary threshold
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
                
            logger.info(f"Complete! Models and results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error in preprocessing or model training: {e}", exc_info=True)
            
    else:
        logger.error("No files found to process.")

if __name__ == "__main__":
    main()