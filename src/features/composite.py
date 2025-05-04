import pandas as pd
from typing import List
from src.features.base import FeatureExtractor 

class CompositeFeatureExtractor(FeatureExtractor):
    """Combines multiple feature extractors."""
    
    def __init__(self, extractors: List[FeatureExtractor]):
        """
        Initialize with a list of feature extractors.
        
        Args:
            extractors: List of FeatureExtractor instances
        """
        self.extractors = extractors
        name = "_".join(e.name for e in extractors)
        super().__init__(name)
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature extractors in sequence."""
        result = df.copy()
        for extractor in self.extractors:
            result = extractor.extract_features(result)
        return result