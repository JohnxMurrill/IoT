import pandas as pd

class FeatureExtractor:
    """Base class for feature extraction from IoT-23 data."""
    
    def __init__(self, name: str):
        """
        Initialize the feature extractor.
        
        Args:
            name: Name of the feature extractor
        """
        self.name = name
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        # Base implementation just returns the input
        return df