import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from typing import List, Tuple, Dict, Optional, Union

logger = logging.getLogger('iot23_framework')

class DataPreprocessor:
    """Handles preprocessing of IoT-23 features for ML models."""
    
    def __init__(self, handle_unknown_categories: str = 'ignore'):
        """
        Initialize the preprocessor.
        
        Args:
            handle_unknown_categories: Strategy for handling unknown categories
                in categorical variables ('ignore' or 'error')
        """
        self.scalers = {}
        self.encoders = {}
        self.label_encoder = None
        self.feature_columns = None
        self.target_column = None
        self.handle_unknown = handle_unknown_categories
        self.fitted = False
        self.categorical_mappings = {}
        self.numerical_stats = {}
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None,
        drop_na: bool = True,
        handle_imbalance: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessors and transform data.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names to scale
            drop_na: Whether to drop rows with NA values
            handle_imbalance: Whether to handle class imbalance
            
        Returns:
            X: Preprocessed features as numpy array
            y: Target variable as numpy array
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to fit_transform")
            return np.array([]), np.array([])
            
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle duration column if it's a string or timedelta
        if 'duration' in df_processed.columns:
            if pd.api.types.is_string_dtype(df_processed['duration']):
                try:
                    df_processed['duration'] = pd.to_timedelta(df_processed['duration']).dt.total_seconds() * 1000
                except:
                    # If conversion fails, try to extract numeric part
                    df_processed['duration'] = df_processed['duration'].str.extract(r'([\d\.]+)').astype(float)
            elif pd.api.types.is_timedelta64_dtype(df_processed['duration']):
                df_processed['duration'] = df_processed['duration'].dt.total_seconds() * 1000

        self.target_column = target_column
        
        # If columns not specified, infer them
        if categorical_columns is None:
            categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            # Remove target if it's in categorical columns
            if target_column in categorical_columns:
                categorical_columns.remove(target_column)
                
        # Ensure all categorical columns exist in the DataFrame
        categorical_columns = [col for col in categorical_columns if col in df_processed.columns]
        logger.info(f"Categorical columns: {categorical_columns}")
        
        if numerical_columns is None:
            numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target if it's in numerical columns
            if target_column in numerical_columns:
                numerical_columns.remove(target_column)
                
        # Ensure all numerical columns exist in the DataFrame
        numerical_columns = [col for col in numerical_columns if col in df_processed.columns]
        logger.info(f"Numerical columns: {numerical_columns}")
        
        # Store feature columns
        self.feature_columns = numerical_columns + categorical_columns
        
        # Handle missing values
        if drop_na:
            columns_to_check = numerical_columns + categorical_columns + [target_column]
            columns_to_check = [col for col in columns_to_check if col in df_processed.columns]
            
            before_len = len(df_processed)
            df_processed = df_processed.dropna(subset=columns_to_check)
            after_len = len(df_processed)
            
            if after_len < before_len:
                logger.info(f"Dropped {before_len - after_len} rows with missing values")
        else:
            # Fill missing values instead of dropping
            for col in numerical_columns:
                if col in df_processed.columns:
                    median_val = df_processed[col].median()
                    df_processed[col] = df_processed[col].fillna(median_val)
                    self.numerical_stats[col] = {'median': median_val}
                    
            for col in categorical_columns:
                if col in df_processed.columns:
                    most_common = df_processed[col].value_counts().index[0] if not df_processed[col].isna().all() else "UNKNOWN"
                    df_processed[col] = df_processed[col].fillna(most_common)
                    self.categorical_mappings[col] = {'missing_value': most_common}
        
        # Extract target
        y = df_processed[target_column].values
        
        # Encode target if it's categorical
        if pd.api.types.is_object_dtype(df_processed[target_column]) or pd.api.types.is_categorical_dtype(df_processed[target_column]):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            logger.info(f"Target classes: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Process numerical features
        numerical_features = None
        if numerical_columns:
            numerical_df = df_processed[numerical_columns].copy()
            
            # Replace infinities with large values
            numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill remaining NaNs with 0
            numerical_df = numerical_df.fillna(0)
            
            # Scale the numerical features
            scaler = StandardScaler()
            numerical_features = scaler.fit_transform(numerical_df)
            self.scalers['standard'] = scaler
        
        # Process categorical features
        categorical_features = None
        if categorical_columns:
            categorical_df = df_processed[categorical_columns].copy()
            
            # Fill NaNs with 'unknown'
            categorical_df = categorical_df.fillna('unknown')
            
            # Convert boolean columns to string
            for col in categorical_df.columns:
                if pd.api.types.is_bool_dtype(categorical_df[col]):
                    categorical_df[col] = categorical_df[col].astype(str)
            
            # One-hot encode
            encoder = OneHotEncoder(sparse_output=False, handle_unknown=self.handle_unknown)
            categorical_features = encoder.fit_transform(categorical_df)
            self.encoders['onehot'] = encoder
            
            # Store feature names for later reference
            self.categorical_feature_names = encoder.get_feature_names_out(categorical_columns)
            logger.info(f"Generated {len(self.categorical_feature_names)} one-hot encoded features")
        
        # Combine features
        if numerical_features is not None and categorical_features is not None:
            X = np.hstack([numerical_features, categorical_features])
        elif numerical_features is not None:
            X = numerical_features
        elif categorical_features is not None:
            X = categorical_features
        else:
            raise ValueError("No features to process")
        
        self.fitted = True
        return X, y
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            X: Preprocessed features as numpy array
            y: Target variable as numpy array (if target column exists)
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        if df.empty:
            logger.warning("Empty DataFrame provided to transform")
            if self.target_column in df.columns:
                return np.array([]), np.array([])
            else:
                return np.array([])
        
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle duration column if it's a string or timedelta
        if 'duration' in df_processed.columns:
            if pd.api.types.is_string_dtype(df_processed['duration']):
                try:
                    df_processed['duration'] = pd.to_timedelta(df_processed['duration']).dt.total_seconds() * 1000
                except:
                    # If conversion fails, try to extract numeric part
                    df_processed['duration'] = df_processed['duration'].str.extract(r'([\d\.]+)').astype(float)
            elif pd.api.types.is_timedelta64_dtype(df_processed['duration']):
                df_processed['duration'] = df_processed['duration'].dt.total_seconds() * 1000
        
        # Extract numerical and categorical columns
        numerical_columns = [col for col in self.feature_columns 
                           if col in df_processed.columns and 
                           col not in self.encoders.get('onehot', {}).feature_names_in_]
        
        categorical_columns = [col for col in self.feature_columns 
                              if col in df_processed.columns and 
                              col in getattr(self.encoders.get('onehot', {}), 'feature_names_in_', [])]
        
        # Process numerical features
        numerical_features = None
        if numerical_columns and 'standard' in self.scalers:
            numerical_df = df_processed[numerical_columns].copy()
            
            # Replace infinities with large values
            numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill remaining NaNs with 0
            numerical_df = numerical_df.fillna(0)
            
            # Scale the numerical features
            numerical_features = self.scalers['standard'].transform(numerical_df)
        
        # Process categorical features
        categorical_features = None
        if categorical_columns and 'onehot' in self.encoders:
            categorical_df = df_processed[categorical_columns].copy()
            
            # Fill NaNs with 'unknown'
            categorical_df = categorical_df.fillna('unknown')
            
            # Convert boolean columns to string
            for col in categorical_df.columns:
                if pd.api.types.is_bool_dtype(categorical_df[col]):
                    categorical_df[col] = categorical_df[col].astype(str)
            
            # One-hot encode
            categorical_features = self.encoders['onehot'].transform(categorical_df)
        
        # Combine features
        if numerical_features is not None and categorical_features is not None:
            X = np.hstack([numerical_features, categorical_features])
        elif numerical_features is not None:
            X = numerical_features
        elif categorical_features is not None:
            X = categorical_features
        else:
            raise ValueError("No features to process")
        
        # Extract target if it exists
        if self.target_column in df_processed.columns:
            y = df_processed[self.target_column].values
            
            # Encode target if it's categorical and we have a label encoder
            if self.label_encoder is not None:
                # Handle unseen classes
                y = np.array([self.label_encoder.transform([cls])[0] if cls in self.label_encoder.classes_ 
                             else -1 for cls in y])
            
            return X, y
        else:
            return X
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the transformed features."""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        feature_names = []
        
        # Add numerical feature names
        numerical_columns = [col for col in self.feature_columns 
                           if col not in self.encoders.get('onehot', {}).feature_names_in_]
        feature_names.extend(numerical_columns)
        
        # Add categorical feature names
        if hasattr(self, 'categorical_feature_names'):
            feature_names.extend(self.categorical_feature_names)
            
        return feature_names