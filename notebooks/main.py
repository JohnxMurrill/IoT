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

if __name__ == "__main__":
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('iot23_framework')
    
    data_dir = Path("C:\\Users\murri\PycharmProjects\IoT\data\data\IoT-23-Max")
    
    # Initialize data loader
    loader = DataLoader(data_dir, chunk_size=100_000)
    
    # Print available files
    files_info = loader.get_files_info()
    print(f"Found {len(files_info)} files:")
    print(files_info)
    
    # Create feature extractors
    network_features = NetworkTrafficFeatures()
    time_features = TimeBasedFeatures()
    
    composite_extractor = CompositeFeatureExtractor([
        network_features,
        time_features
    ])
    
    # Process a sample file
    if len(loader.file_paths) > 0:
        sample_files = loader.file_paths[4]  # Adjust as needed
        print(f"Processing sample files: {sample_files}")
        
        processed_data = loader.process_file(
            sample_files,
            feature_extractor=composite_extractor,
            cache_key=f"{sample_files.stem}_features"
        )
        
        print(f"Processed data shape: {processed_data.shape}")
        print("Feature columns:")
        print(processed_data.columns.tolist())
        
        # Preprocess features - encodings, conversions, etc
        preprocessor = DataPreprocessor()
        
        target_col = 'label' if 'label' in processed_data.columns else 'label'
        
        # Check if target column exists, otherwise use a dummy
        if target_col not in processed_data.columns:
            print(f"Warning: Target column '{target_col}' not found. Creating dummy for example.")
            processed_data[target_col] = 'normal'  # Dummy label

        # Fit and transform
        categorical_columns = [
            'proto',
            'service',
            'conn_state',
            'local_orig',
            'local_resp',
            'label'
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
        X, y = preprocessor.fit_transform(processed_data, target_col, categorical_columns=categorical_columns, numerical_columns=numerical_columns)
        print(f"Preprocessed X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        if 'timestamp' in processed_data.columns:
            print("Performing time-based split")
            train_df, val_df, test_df = DataSplitter.time_based_split(
                processed_data, 'timestamp'
            )
        else:
            print("Performing stratified split")
            train_df, val_df, test_df = DataSplitter.stratified_split(
                processed_data, target_col
            )
        
        print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")