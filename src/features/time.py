import logging
import pandas as pd
import numpy as np
from src.features.base import FeatureExtractor

logger = logging.getLogger('iot23_framework')

class TimeBasedFeatures(FeatureExtractor):
    """Extract time-based features from IoT-23 data."""
    
    def __init__(self):
        super().__init__("time_based")
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features."""
        features = df.copy()
        
        try:
            # Convert timestamp to datetime if needed
            # Adjust 'ts' to match the actual timestamp column in IoT-23
            if 'ts' in features.columns:
                if not pd.api.types.is_datetime64_any_dtype(features['ts']):
                    features['timestamp'] = pd.to_datetime(features['ts'], unit='s')
                else:
                    features['timestamp'] = features['ts']
                
                # Extract time components
                features['hour'] = features['timestamp'].dt.hour
                features['day_of_week'] = features['timestamp'].dt.dayofweek
                features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
                
                # Create cyclical time features
                features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
                features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
                features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
                
        except Exception as e:
            logger.error(f"Error extracting time features: {e}")
            # Return original data if feature extraction fails
            return df
        
        return features