import sys
import logging
from pathlib import Path

root_dir = Path(__file__).parent.parent
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

def main():
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('iot23_framework')
    
    # Data directory - adjust to your path
    data_dir = Path("C:\\Users\\murri\\PycharmProjects\\IoT\\data\\data\\IoT-23-Max")
    
    # Initialize data loader
    loader = DataLoader(data_dir, chunk_size=100_000)
    
    # Print available files
    files_info = loader.get_files_info()
    logger.info(f"Found {len(files_info)} files:")
    logger.info(files_info)
    
    # Create feature extractors
    network_features = NetworkTrafficFeatures()
    time_features = TimeBasedFeatures()
    
    # Create composite extractor
    composite_extractor = CompositeFeatureExtractor([
        network_features,
        time_features
    ])
    
    # Process a sample file
    if len(loader.file_paths) > 0:
        sample_files = loader.file_paths[4]  # Adjust as needed
        logger.info(f"Processing sample files: {sample_files}")
        
        # Process with feature extraction
        processed_data = loader.process_file(
            sample_files,
            feature_extractor=composite_extractor,
            cache_key=f"{sample_files.stem}_features"
        )
        
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Feature columns: {processed_data.columns.tolist()}")
        
        # Preprocess for ML
        preprocessor = DataPreprocessor()
        
        # Assuming 'label' is the target column - adjust for actual IoT-23 format
        target_col = 'label' if 'label' in processed_data.columns else 'label'
        
        # Check if target column exists, otherwise use a dummy
        if target_col not in processed_data.columns:
            logger.warning(f"Target column '{target_col}' not found. Creating dummy for example.")
            processed_data[target_col] = 'normal'  # Dummy label

        # Define feature columns
        categorical_columns = [
            'proto',
            'service',
            'conn_state',
            'local_orig',
            'local_resp'
        ]
        numerical_columns = [
            'id.orig_p',
            'id.resp_p',
            'duration',
            'orig_bytes',
            'resp_bytes',
            'missed_bytes',
            'orig_pkts',
            'orig_ip_bytes',
            'resp_pkts',
            'resp_ip_bytes'
        ]
        
        # Add any custom features from your extractors
        # Example: numerical_columns.extend(['connection_rate', 'packet_rate'])
        
        # Fit and transform
        X, y = preprocessor.fit_transform(
            processed_data, 
            target_col, 
            categorical_columns=categorical_columns, 
            numerical_columns=numerical_columns
        )
        logger.info(f"Preprocessed X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
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
        X_train, y_train = preprocessor.transform(train_df, target_col)
        X_val, y_val = preprocessor.transform(val_df, target_col)
        X_test, y_test = preprocessor.transform(test_df, target_col)
        
        logger.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
        logger.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")
        logger.info(f"Test data shapes - X: {X_test.shape}, y: {y_test.shape}")
        
        # Create model instances
        models = {
            'random_forest': RandomForestModel(n_estimators=100, max_depth=10),
            'svm': SVMModel(kernel='rbf', C=1.0),
            'knn': KNNModel(n_neighbors=5),
            'gaussian_nb': GaussianNBModel(),
            'logistic_regression': LogisticRegressionModel(),
            'gmm': GMMClassifier(n_components=5)
        }
        
        # Initialize model trainer
        trainer = ModelTrainer(models, output_dir=Path("models"))
        
        # Train all models
        logger.info("Training models...")
        trainer.train_all(X_train, y_train)
        
        # Optional: Hyperparameter tuning example for Random Forest
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
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

if __name__ == "__main__":
    main()