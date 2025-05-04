import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple

class DataPreprocessor:
    """Handles preprocessing of IoT-23 features for ML models."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = None
        self.target_column = None
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessors and transform data.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names to scale
            
        Returns:
            X: Preprocessed features as numpy array
            y: Target variable as numpy array
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        self.target_column = target_column
        
        # If columns not specified, infer them
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Remove target if it's in categorical columns
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
        print(f"Categorical columns: {categorical_columns}")
        print(f"Numerical columns: {numerical_columns}")
        
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target if it's in numerical columns
            if target_column in numerical_columns:
                numerical_columns.remove(target_column)
        
        # Store feature columns
        self.feature_columns = numerical_columns + categorical_columns
        
        # Extract target
        y = df[target_column].values
        
        # Process numerical features
        numerical_features = None
        if numerical_columns:
            scaler = StandardScaler()
            print(df.columns)
            numerical_features = scaler.fit_transform(df[numerical_columns].fillna(0))
            self.scalers['standard'] = scaler
        
        # Process categorical features
        categorical_features = None
        if categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_features = encoder.fit_transform(df[categorical_columns].fillna('unknown'))
            self.encoders['onehot'] = encoder
        
        # Combine features
        if numerical_features is not None and categorical_features is not None:
            X = np.hstack([numerical_features, categorical_features])
        elif numerical_features is not None:
            X = numerical_features
        elif categorical_features is not None:
            X = categorical_features
        else:
            raise ValueError("No features to process")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            X: Preprocessed features as numpy array
        """
        if not self.feature_columns:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        numerical_columns = [col for col in self.feature_columns 
                            if col in df.columns and col not in self.encoders.get('onehot', {}).get('feature_names_in_', [])]
        
        categorical_columns = [col for col in self.feature_columns 
                              if col in df.columns and col in self.encoders.get('onehot', {}).get('feature_names_in_', [])]
        
        # Process numerical features
        numerical_features = None
        if numerical_columns and 'standard' in self.scalers:
            numerical_features = self.scalers['standard'].transform(df[numerical_columns].fillna(0))
        
        # Process categorical features
        categorical_features = None
        if categorical_columns and 'onehot' in self.encoders:
            categorical_features = self.encoders['onehot'].transform(df[categorical_columns].fillna('unknown'))
        
        # Combine features
        if numerical_features is not None and categorical_features is not None:
            X = np.hstack([numerical_features, categorical_features])
        elif numerical_features is not None:
            X = numerical_features
        elif categorical_features is not None:
            X = categorical_features
        else:
            raise ValueError("No features to process")
        
        return X