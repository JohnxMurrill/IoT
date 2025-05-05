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
        drop_na: bool = False,  # Changed default to False
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
        
        # Print initial dataset size
        logger.info(f"Initial dataset size: {len(df_processed)} rows")
        
        # Normalize labels in target column if it's string type
        if pd.api.types.is_object_dtype(df_processed[target_column]):
            # Convert to lowercase
            df_processed[target_column] = df_processed[target_column].str.lower()
            
            # Standardize common variations
            label_mapping = {
                'malicious': 'malicious',
                'benign': 'benign', 
                'normal': 'benign',
                'anomaly': 'malicious',
                'attack': 'malicious'
            }
            
            # Apply mapping
            df_processed[target_column] = df_processed[target_column].map(
                lambda x: label_mapping.get(x.lower() if isinstance(x, str) else x, x) 
                if x is not None else 'unknown'
            )
            
            # Log unique class values
            logger.info(f"Target classes after normalization: {df_processed[target_column].unique()}")
        
        # Handle duration column if it's a string or timedelta
        if 'duration' in df_processed.columns:
            if pd.api.types.is_string_dtype(df_processed['duration']):
                try:
                    df_processed['duration'] = pd.to_timedelta(df_processed['duration']).dt.total_seconds() * 1000
                except:
                    # If conversion fails, try to extract numeric part
                    df_processed['duration'] = pd.to_numeric(
                        df_processed['duration'].str.extract(r'([\d\.]+)')[0], 
                        errors='coerce'
                    ).fillna(0)
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
        required_columns = [target_column]  # Only require target column
        
        # Count NA values before handling
        na_counts = {}
        for col in df_processed.columns:
            na_count = df_processed[col].isna().sum()
            if na_count > 0:
                na_counts[col] = na_count
                
        if na_counts:
            logger.info(f"NA counts before handling: {na_counts}")
            
        if drop_na:
            # Only drop rows with NA in required columns
            before_len = len(df_processed)
            df_processed = df_processed.dropna(subset=required_columns)
            after_len = len(df_processed)
            
            if after_len < before_len:
                logger.info(f"Dropped {before_len - after_len} rows with missing values in required columns")
                
            # Check if we still have data
            if len(df_processed) == 0:
                logger.error("All rows were dropped! Switching to filling NA values instead.")
                # Restore the original data
                df_processed = df.copy()
                drop_na = False
        
        # Fill missing values for remaining columns
        for col in numerical_columns:
            if col in df_processed.columns and df_processed[col].isna().any():
                fill_value = df_processed[col].median() if not pd.isna(df_processed[col].median()) else 0
                df_processed[col] = df_processed[col].fillna(fill_value)
                self.numerical_stats[col] = {'median': fill_value}
                logger.info(f"Filled NA values in '{col}' with median: {fill_value}")
                
        # Fill missing values for remaining columns
        for col in categorical_columns:
            if col in df_processed.columns and df_processed[col].isna().any():
                # Special handling for boolean columns
                if pd.api.types.is_bool_dtype(df_processed[col]):
                    # Use the most common boolean value or False as default
                    most_common = df_processed[col].mode()[0] if not df_processed[col].isna().all() else False
                    df_processed[col] = df_processed[col].fillna(most_common)
                    self.categorical_mappings[col] = {'missing_value': most_common}
                    logger.info(f"Filled NA values in boolean column '{col}' with: {most_common}")
                else:
                    # For non-boolean categorical columns, use mode or 'unknown'
                    most_common = df_processed[col].mode()[0] if not df_processed[col].isna().all() else "unknown"
                    df_processed[col] = df_processed[col].fillna(most_common)
                    self.categorical_mappings[col] = {'missing_value': most_common}
                    logger.info(f"Filled NA values in '{col}' with mode: {most_common}")
                
        # Fill any remaining NAs in target column
        if df_processed[target_column].isna().any():
            logger.warning(f"Found NA values in target column '{target_column}'. Filling with 'unknown'")
            df_processed[target_column] = df_processed[target_column].fillna('unknown')
        
        # Log dataset size after handling NAs
        logger.info(f"Dataset size after handling NAs: {len(df_processed)} rows")
        
        # Verify we still have data
        if len(df_processed) == 0:
            logger.error("No data left after preprocessing! Returning empty arrays.")
            return np.array([]), np.array([])
        
        # Extract target
        y = df_processed[target_column].values
        
        # Encode target if it's categorical
        if pd.api.types.is_object_dtype(df_processed[target_column]) or pd.api.types.is_categorical_dtype(df_processed[target_column]):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            logger.info(f"Target classes encoded: {list(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Process numerical features
        numerical_features = None
        if numerical_columns and len(df_processed) > 0:
            numerical_df = df_processed[numerical_columns].copy()
            
            # Replace infinities with large values
            numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill remaining NaNs with 0
            numerical_df = numerical_df.fillna(0)
            
            # Verify numerical data
            for col in numerical_df.columns:
                if not pd.api.types.is_numeric_dtype(numerical_df[col]):
                    logger.warning(f"Column '{col}' is not numeric. Converting to numeric.")
                    numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce').fillna(0)
            
            # Scale the numerical features
            scaler = StandardScaler()
            try:
                numerical_features = scaler.fit_transform(numerical_df)
                self.scalers['standard'] = scaler
            except Exception as e:
                logger.error(f"Error scaling numerical features: {e}")
                # Fall back to unscaled features
                numerical_features = numerical_df.values
        
        # Process categorical features
        categorical_features = None
        if categorical_columns and len(df_processed) > 0:
            categorical_df = df_processed[categorical_columns].copy()
            
            # Fill NaNs with 'unknown'
            categorical_df = categorical_df.fillna('unknown')
            
            # Convert boolean columns to string
            for col in categorical_df.columns:
                if pd.api.types.is_bool_dtype(categorical_df[col]):
                    categorical_df[col] = categorical_df[col].astype(str)
            
            # One-hot encode
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown=self.handle_unknown)
                categorical_features = encoder.fit_transform(categorical_df)
                self.encoders['onehot'] = encoder
                
                # Store feature names for later reference
                self.categorical_feature_names = encoder.get_feature_names_out(categorical_columns)
                logger.info(f"Generated {len(self.categorical_feature_names)} one-hot encoded features")
            except Exception as e:
                logger.error(f"Error one-hot encoding categorical features: {e}")
                # Fall back to dummy variables
                try:
                    dummies = pd.get_dummies(categorical_df, dummy_na=True)
                    categorical_features = dummies.values
                    logger.info(f"Used pandas get_dummies as fallback")
                except Exception as e2:
                    logger.error(f"Error creating dummy variables: {e2}")
                    # Return with just numerical features if we have them
                    if numerical_features is not None:
                        self.fitted = True
                        return numerical_features, y
                    else:
                        # We have no features at all
                        logger.error("Failed to create any features!")
                        return np.array([[]]), y
        
        # Combine features
        if numerical_features is not None and categorical_features is not None:
            X = np.hstack([numerical_features, categorical_features])
        elif numerical_features is not None:
            X = numerical_features
        elif categorical_features is not None:
            X = categorical_features
        else:
            logger.error("No features to process")
            return np.array([[]]), y
        
        logger.info(f"Final preprocessed data shape: X={X.shape}, y={y.shape}")
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
                return np.array([[]]), np.array([])
            else:
                return np.array([[]])
        
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Normalize target if present
        if self.target_column in df_processed.columns and pd.api.types.is_object_dtype(df_processed[self.target_column]):
            # Convert to lowercase
            df_processed[self.target_column] = df_processed[self.target_column].str.lower()
            
            # Standardize common variations
            label_mapping = {
                'malicious': 'malicious',
                'benign': 'benign', 
                'normal': 'benign',
                'anomaly': 'malicious',
                'attack': 'malicious'
            }
            
            # Apply mapping
            df_processed[self.target_column] = df_processed[self.target_column].map(
                lambda x: label_mapping.get(x.lower() if isinstance(x, str) else x, x) 
                if x is not None else 'unknown'
            )
        
        # Handle duration column if it's a string or timedelta
        if 'duration' in df_processed.columns:
            if pd.api.types.is_string_dtype(df_processed['duration']):
                try:
                    df_processed['duration'] = pd.to_timedelta(df_processed['duration']).dt.total_seconds() * 1000
                except:
                    # If conversion fails, try to extract numeric part
                    df_processed['duration'] = pd.to_numeric(
                        df_processed['duration'].str.extract(r'([\d\.]+)')[0], 
                        errors='coerce'
                    ).fillna(0)
            elif pd.api.types.is_timedelta64_dtype(df_processed['duration']):
                df_processed['duration'] = df_processed['duration'].dt.total_seconds() * 1000
        
        # Extract numerical and categorical columns
        numerical_columns = [col for col in self.feature_columns 
                           if col in df_processed.columns and 
                           col not in getattr(self.encoders.get('onehot', {}), 'feature_names_in_', [])]
        
        categorical_columns = [col for col in self.feature_columns 
                              if col in df_processed.columns and 
                              col in getattr(self.encoders.get('onehot', {}), 'feature_names_in_', [])]
        
        # Fill missing values
        for col in numerical_columns:
            if col in df_processed.columns and df_processed[col].isna().any():
                fill_value = self.numerical_stats.get(col, {}).get('median', 0)
                df_processed[col] = df_processed[col].fillna(fill_value)
                
        for col in categorical_columns:
            if col in df_processed.columns and df_processed[col].isna().any():
                fill_value = self.categorical_mappings.get(col, {}).get('missing_value', 'unknown')
                df_processed[col] = df_processed[col].fillna(fill_value)
        
        # Process numerical features
        numerical_features = None
        if numerical_columns and 'standard' in self.scalers:
            numerical_df = df_processed[numerical_columns].copy()
            
            # Replace infinities with large values
            numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill remaining NaNs with 0
            numerical_df = numerical_df.fillna(0)
            
            # Verify numerical data
            for col in numerical_df.columns:
                if not pd.api.types.is_numeric_dtype(numerical_df[col]):
                    numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce').fillna(0)
            
            try:
                # Scale the numerical features
                numerical_features = self.scalers['standard'].transform(numerical_df)
            except Exception as e:
                logger.error(f"Error scaling numerical features in transform: {e}")
                # Fall back to unscaled features
                numerical_features = numerical_df.values
        
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
            
            try:
                # One-hot encode
                categorical_features = self.encoders['onehot'].transform(categorical_df)
            except Exception as e:
                logger.error(f"Error one-hot encoding categorical features in transform: {e}")
                # Fall back to dummy variables
                try:
                    dummies = pd.get_dummies(categorical_df, dummy_na=True)
                    # Make sure we have the same columns as during training
                    expected_cols = self.categorical_feature_names
                    # Create a DataFrame with all expected columns filled with zeros
                    result = pd.DataFrame(0, index=range(len(dummies)), columns=expected_cols)
                    # Fill in the values we have
                    for col in dummies.columns:
                        if col in expected_cols:
                            result[col] = dummies[col]
                    categorical_features = result.values
                except Exception as e2:
                    logger.error(f"Error creating dummy variables in transform: {e2}")
                    # Return with just numerical features if we have them
                    if numerical_features is not None:
                        if self.target_column in df_processed.columns:
                            y = df_processed[self.target_column].values
                            if self.label_encoder is not None:
                                # Handle unknown labels
                                y = np.array([
                                    self.label_encoder.transform([label])[0] 
                                    if label in self.label_encoder.classes_ else -1 
                                    for label in y
                                ])
                            return numerical_features, y
                        else:
                            return numerical_features
                    else:
                        # We have no features at all
                        logger.error("Failed to create any features in transform!")
                        if self.target_column in df_processed.columns:
                            return np.array([[]]), df_processed[self.target_column].values
                        else:
                            return np.array([[]])
        
        # Combine features
        if numerical_features is not None and categorical_features is not None:
            X = np.hstack([numerical_features, categorical_features])
        elif numerical_features is not None:
            X = numerical_features
        elif categorical_features is not None:
            X = categorical_features
        else:
            logger.error("No features created in transform!")
            X = np.array([[]])
        
        # Extract target if it exists
        if self.target_column in df_processed.columns:
            y = df_processed[self.target_column].values
            
            # Encode target if it's categorical and we have a label encoder
            if self.label_encoder is not None:
                # Handle unseen classes
                y = np.array([
                    self.label_encoder.transform([label])[0] 
                    if label in self.label_encoder.classes_ else -1 
                    for label in y
                ])
            
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
                           if col not in getattr(self.encoders.get('onehot', {}), 'feature_names_in_', [])]
        feature_names.extend(numerical_columns)
        
        # Add categorical feature names
        if hasattr(self, 'categorical_feature_names'):
            feature_names.extend(self.categorical_feature_names)
            
        return feature_names