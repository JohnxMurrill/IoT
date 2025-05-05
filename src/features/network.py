from src.features.base import FeatureExtractor
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('iot23_framework')

class NetworkTrafficFeatures(FeatureExtractor):
    """Extract network traffic features from IoT-23 data."""
    
    def __init__(self):
        super().__init__("network_traffic")
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract network traffic features."""
        # Create a copy to avoid modifying the original
        features = df.copy()
        
        # Basic features - adjust column names based on actual IoT-23 format
        try:
            # Convert string columns to numeric if needed
            numeric_columns = ['orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'duration']
            for col in numeric_columns:
                if col in features.columns and pd.api.types.is_string_dtype(features[col]):
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                    
            # Extract packet size features
            if 'orig_bytes' in features.columns and 'resp_bytes' in features.columns:
                features['total_bytes'] = features['orig_bytes'] + features['resp_bytes']
                features['bytes_ratio'] = features['orig_bytes'] / (features['resp_bytes'] + 1)
            
            # Extract packet count features
            if 'orig_pkts' in features.columns and 'resp_pkts' in features.columns:
                features['total_pkts'] = features['orig_pkts'] + features['resp_pkts']
                features['pkts_ratio'] = features['orig_pkts'] / (features['resp_pkts'] + 1)
            
            # Duration and rate features
            if 'duration' in features.columns:
                # Avoid division by zero
                safe_duration = features['duration'].replace(0, np.nan)
                
                if 'total_bytes' in features.columns:
                    features['bytes_per_sec'] = features['total_bytes'] / safe_duration
                
                if 'total_pkts' in features.columns:
                    features['pkts_per_sec'] = features['total_pkts'] / safe_duration
            
            # Protocol features
            if 'proto' in features.columns:
                # One-hot encode protocol
                proto_dummies = pd.get_dummies(features['proto'], prefix='proto')
                features = pd.concat([features, proto_dummies], axis=1)
            
            # Clean up NaN values
            for col in features.select_dtypes(include=[np.number]).columns:
                features[col] = features[col].fillna(0)
            
        except Exception as e:
            logger.error(f"Error extracting network features: {e}")
            # Return original data if feature extraction fails
            return df
        
        return features
