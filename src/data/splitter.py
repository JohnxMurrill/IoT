import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

logger = logging.getLogger('iot23_framework')

class DataSplitter:
    """Handles train/val/test splitting with respect to time series nature."""
    
    @staticmethod
    def time_based_split(
        df: pd.DataFrame, 
        timestamp_col: str,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by time to prevent future data leakage.
        
        Args:
            df: Input DataFrame
            timestamp_col: Column containing timestamps
            val_size: Proportion for validation
            test_size: Proportion for testing
            
        Returns:
            train_df, val_df, test_df: Split DataFrames
        """
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(by=timestamp_col)
        
        # Calculate split indices
        n = len(df)
        test_start_idx = int(n * (1 - test_size))
        val_start_idx = int(test_start_idx * (1 - val_size))
        
        # Split data
        train_df = df.iloc[:val_start_idx]
        val_df = df.iloc[val_start_idx:test_start_idx]
        test_df = df.iloc[test_start_idx:]
        
        logger.info(f"Time-based split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def stratified_split(
        df: pd.DataFrame,
        target_col: str,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data with stratification by target class.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_state: Random seed
            
        Returns:
            train_df, val_df, test_df: Split DataFrames
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            stratify=df[target_col],
            random_state=random_state
        )
        
        # Second split: train vs val
        # Recalculate val_size as proportion of train_val
        adj_val_size = val_size / (1 - test_size)
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adj_val_size,
            stratify=train_val_df[target_col],
            random_state=random_state
        )
        
        logger.info(f"Stratified split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df