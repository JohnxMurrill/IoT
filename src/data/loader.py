import pandas as pd
import numpy as np
import logging
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from utils.caching import CACHE_DIR
from typing import List, Generator, Optional, Union, Dict

logger = logging.getLogger('iot23_framework')

DTYPES = {
    'uid': 'string',
    'id.orig_h': 'string',
    'id.orig_p': 'int32',
    'id.resp_h': 'string',
    'id.resp_p': 'int32',
    'proto': 'string',
    'service': 'string',
    'duration': 'string',
    'orig_bytes': 'float32',
    'resp_bytes': 'float32',
    'conn_state': 'string',
    'local_orig': 'boolean',
    'local_resp': 'boolean',
    'missed_bytes': 'float32',
    'history': 'string',
    'orig_pkts': 'float32',
    'orig_ip_bytes': 'float32',
    'resp_pkts': 'float32',
    'resp_ip_bytes': 'float32',
    'tunnel_parents': 'string', 
    'label': 'string',
    'detailed-label': 'string',
}

class DataLoader:
    """
    Handles efficient loading of large IoT-23 dataset files.
    Implements chunking and parallel processing to handle data 
    that doesn't fit in memory.
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        chunk_size: int = 500_000,
        n_jobs: int = -1,
        cache_processed: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the IoT-23 dataset files
            chunk_size: Number of rows to process at once
            n_jobs: Number of parallel jobs (-1 uses all cores)
            cache_processed: Whether to cache processed data
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.cache_processed = cache_processed
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.file_paths = self._discover_files()
        logger.info(f"Found {len(self.file_paths)} data files")
    
    def _discover_files(self) -> List[Path]:
        """Find all relevant data files in the data directory."""
        return list(self.data_dir.glob("*.csv"))
    
    def get_files_info(self) -> pd.DataFrame:
        """Returns information about available data files."""
        info = []
        for path in self.file_paths:
            # Get basic file info
            size_mb = path.stat().st_size / (1024 * 1024)
            # Try to determine if benign or malicious from filename
            is_malicious = "malicious" in path.name.lower()
            is_benign = "benign" in path.name.lower()
            category = "malicious" if is_malicious else ("benign" if is_benign else "unknown")
            
            info.append({
                "filename": path.name,
                "path": str(path),
                "size_mb": round(size_mb, 2),
                "category": category
            })
        
        return pd.DataFrame(info)
    
    def load_file_chunks(
        self, 
        file_path: Union[str, Path],
        columns: Optional[List[str]] = None
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Generator to load and yield chunks of a large file.
        
        Args:
            file_path: Path to the data file
            columns: List of columns to load (None for all)
            
        Yields:
            DataFrame chunks
        """
        file_path = Path(file_path)
        
        logger.info(f"Loading file in chunks: {file_path.name}")
        try:
            if columns is None:
                reader = pd.read_csv(
                    file_path,
                    header=0,
                    chunksize=self.chunk_size,
                    low_memory=False,
                    on_bad_lines='warn',
                    dtype={col: DTYPES.get(col, 'object') for col in DTYPES.keys()}
                )
            else:
                reader = pd.read_csv(
                    file_path,
                    header=None,
                    names=columns,
                    chunksize=self.chunk_size,
                    low_memory=False,
                    on_bad_lines='warn',
                    dtype={col: DTYPES.get(col, 'object') for col in columns if col in DTYPES}
                )
                
            for chunk in reader:
                if columns is None:
                    chunk.columns = [col.strip() for col in chunk.columns]
                
                # Add or normalize label column
                if 'label' not in chunk.columns:
                    # Extract label from filename
                    filename = file_path.name.lower()
                    if 'benign' in filename:
                        chunk['label'] = 'benign'  # Always lowercase
                    elif 'malicious' in filename:
                        chunk['label'] = 'malicious'  # Always lowercase
                    else:
                        # Try to extract more specific labels
                        if any(attack in filename for attack in ['botnet', 'c&c', 'ddos', 'scan']):
                            for attack in ['botnet', 'c&c', 'ddos', 'scan']:
                                if attack in filename:
                                    chunk['label'] = attack  # Always lowercase
                                    break
                        else:
                            chunk['label'] = 'unknown'
                else:
                    # Normalize existing label values (convert to lowercase)
                    if pd.api.types.is_object_dtype(chunk['label']):
                        # Check original values for debugging
                        unique_before = chunk['label'].unique()
                        
                        # Convert all labels to lowercase for consistency
                        chunk['label'] = chunk['label'].str.lower()
                        
                        # Check values after lowercase conversion
                        unique_after = chunk['label'].unique()
                        
                        # Log if case normalization made a difference
                        if len(unique_before) != len(unique_after):
                            logger.info(f"Label case normalization reduced unique values from {len(unique_before)} to {len(unique_after)}")
                            logger.info(f"Before: {unique_before}")
                            logger.info(f"After: {unique_after}")
                        
                        # Map specific variations to standard forms
                        label_mapping = {
                            'malicious': 'malicious',
                            'benign': 'benign', 
                            'normal': 'benign',
                            'anomaly': 'malicious',
                            'attack': 'malicious'
                        }
                        
                        # Apply mapping to standardize labels
                        chunk['label'] = chunk['label'].map(lambda x: label_mapping.get(x.lower(), x.lower()) if isinstance(x, str) else 'unknown')
                
                # Log data info for debugging
                logger.debug(f"Chunk from {file_path.name}: {len(chunk)} rows, {len(chunk.columns)} columns")
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            # Return empty DataFrame if file can't be loaded
            yield pd.DataFrame()
    
    @staticmethod
    def _process_chunk(chunk: pd.DataFrame, feature_extractor) -> pd.DataFrame:
        """Process a single chunk with feature extraction."""
        if chunk.empty:
            return chunk
        return feature_extractor.extract_features(chunk)
    
    def process_file(
        self,
        file_path: Union[str, Path],
        feature_extractor,
        cache_key: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process an entire file with chunking and optional caching.
        
        Args:
            file_path: Path to the data file
            feature_extractor: FeatureExtractor instance
            cache_key: Key for caching (if None, no caching)
            max_chunks: Maximum number of chunks to process (None for all)
            
        Returns:
            DataFrame with processed features
        """
        file_path = Path(file_path)
        
        # Generate cache key if not provided
        if cache_key is None and self.cache_processed:
            if hasattr(feature_extractor, 'name'):
                extractor_name = feature_extractor.name
            else:
                extractor_name = str(hash(feature_extractor))
            cache_key = f"{file_path.stem}_{extractor_name}"
        
        # Check cache
        cache_path = CACHE_DIR / f"{cache_key}.parquet"
        if self.cache_processed and cache_path.exists():
            logger.info(f"Loading processed data from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Process in chunks
        processed_chunks = []
        for i, chunk in enumerate(self.load_file_chunks(file_path)):
            if max_chunks is not None and i >= max_chunks:
                logger.info(f"Reached max chunks ({max_chunks}) for {file_path.name}")
                break
                
            logger.info(f"Processing chunk {i+1} from {file_path.name}")
            if not chunk.empty:
                processed = self._process_chunk(chunk, feature_extractor)
                processed_chunks.append(processed)
        
        # Combine processed chunks
        if processed_chunks:
            result = pd.concat(processed_chunks, ignore_index=True)
        else:
            logger.warning(f"No data processed from {file_path.name}")
            result = pd.DataFrame()

        # Cache result
        if self.cache_processed and not result.empty:
            logger.info(f"Caching processed data: {cache_path}")
            result.to_parquet(cache_path)
        
        return result
    
    def process_files_parallel(
        self,
        file_paths: List[Union[str, Path]],
        feature_extractor,
        sample_fraction: Optional[float] = None,
        max_chunks_per_file: Optional[int] = None,
        max_rows_total: Optional[int] = None,
        balance_classes: bool = False
    ) -> pd.DataFrame:
        """
        Process multiple files in parallel.
        
        Args:
            file_paths: List of paths to data files
            feature_extractor: FeatureExtractor instance
            sample_fraction: Fraction of data to sample (None for all)
            max_chunks_per_file: Maximum chunks to process per file
            max_rows_total: Maximum total rows in final dataset
            balance_classes: Whether to balance classes in the final dataset
            
        Returns:
            DataFrame with processed features from all files
        """
        file_paths = [Path(p) for p in file_paths]
        
        logger.info(f"Processing {len(file_paths)} files in parallel")
        
        # Create cache keys
        cache_keys = {}
        for fp in file_paths:
            if hasattr(feature_extractor, 'name'):
                extractor_name = feature_extractor.name
            else:
                extractor_name = str(hash(feature_extractor))
                
            cache_keys[fp] = f"{fp.stem}_{extractor_name}"
        
        # Process files in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_file)(
                file_path=fp,
                feature_extractor=feature_extractor,
                cache_key=cache_keys[fp],
                max_chunks=max_chunks_per_file
            )
            for fp in file_paths
        )
        
        # Filter out empty results
        results = [df for df in results if not df.empty]
        
        if not results:
            logger.warning("No data was processed from any files")
            return pd.DataFrame()
        
        # Before combining, ensure consistent label capitalization
        for i, df in enumerate(results):
            if 'label' in df.columns and pd.api.types.is_object_dtype(df['label']):
                # Convert all labels to lowercase
                results[i]['label'] = df['label'].str.lower()
                
                # Map variations to standard forms
                label_mapping = {
                    'malicious': 'malicious',
                    'benign': 'benign', 
                    'normal': 'benign',
                    'anomaly': 'malicious',
                    'attack': 'malicious'
                }
                
                # Apply mapping
                results[i]['label'] = results[i]['label'].map(lambda x: label_mapping.get(x, x) if isinstance(x, str) else 'unknown')
        
        # Combine results
        logger.info("Combining results from all files")
        
        # If balance_classes is True, try to balance the dataset by class
        if balance_classes and all('label' in df.columns for df in results):
            balanced_results = []
            
            # First, count classes across all files
            all_labels = []
            for df in results:
                all_labels.extend(df['label'].value_counts().to_dict().items())
                
            label_counts = {}
            for label, count in all_labels:
                if label in label_counts:
                    label_counts[label] += count
                else:
                    label_counts[label] = count
                    
            # Log original class distribution
            logger.info(f"Original class distribution: {label_counts}")
            
            # Find the minority class count
            if label_counts:
                min_count = min(label_counts.values())
                # Adjust min_count if max_rows_total is specified
                if max_rows_total:
                    samples_per_class = max_rows_total // len(label_counts)
                    min_count = min(min_count, samples_per_class)
                    
                logger.info(f"Balancing classes. Minimum class count: {min_count}")
                
                # Sample each class equally
                for df in results:
                    class_dfs = []
                    for label in df['label'].unique():
                        class_df = df[df['label'] == label]
                        # Calculate how many samples to take from this file for this class
                        # Distribute samples evenly across files
                        class_sample = min(len(class_df), min_count // len(file_paths) + 1)
                        if class_sample > 0:
                            sampled = class_df.sample(class_sample, random_state=42)
                            class_dfs.append(sampled)
                            logger.info(f"Sampled {class_sample} instances of class '{label}' from {df['label'].nunique()} classes")
                    
                    if class_dfs:
                        balanced_results.append(pd.concat(class_dfs, ignore_index=True))
                        
                if balanced_results:
                    combined = pd.concat(balanced_results, ignore_index=True)
                    logger.info(f"Class distribution after balancing: {combined['label'].value_counts().to_dict()}")
                else:
                    # Fallback to simple concat if balancing fails
                    logger.warning("Class balancing failed, using simple concatenation instead")
                    combined = pd.concat(results, ignore_index=True)
            else:
                # Fallback to simple concat if no labels
                logger.warning("No labels found for balancing, using simple concatenation")
                combined = pd.concat(results, ignore_index=True)
        else:
            # Simple concatenation if not balancing
            logger.info("Using simple concatenation of all results")
            combined = pd.concat(results, ignore_index=True)
            
        # Apply max_rows_total if specified
        if max_rows_total and len(combined) > max_rows_total:
            logger.info(f"Sampling dataset to {max_rows_total} rows (from {len(combined)} rows)")
            # Try to maintain class proportions when sampling
            if 'label' in combined.columns:
                combined = combined.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(
                        n=max(1, int(max_rows_total * len(x) / len(combined))),
                        random_state=42
                    )
                )
                # If we still have too many rows (due to rounding), take a final sample
                if len(combined) > max_rows_total:
                    combined = combined.sample(max_rows_total, random_state=42)
            else:
                combined = combined.sample(max_rows_total, random_state=42)
        
        # Apply sample_fraction if specified and max_rows_total not used
        elif sample_fraction is not None and 0 < sample_fraction < 1 and (max_rows_total is None):
            sample_size = int(len(combined) * sample_fraction)
            logger.info(f"Sampling {sample_fraction*100}% of data ({sample_size} rows from {len(combined)})")
            # Try to maintain class proportions when sampling
            if 'label' in combined.columns:
                combined = combined.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(
                        n=max(1, int(sample_size * len(x) / len(combined))),
                        random_state=42
                    )
                )
                # If we still have too many rows (due to rounding), take a final sample
                if len(combined) > sample_size:
                    combined = combined.sample(sample_size, random_state=42)
            else:
                combined = combined.sample(sample_size, random_state=42)
                
        # Final logging of class distribution (if applicable)
        if 'label' in combined.columns:
            final_dist = combined['label'].value_counts().to_dict()
            logger.info(f"Final class distribution: {final_dist}")
            
        logger.info(f"Final dataset shape: {combined.shape}")
        
        return combined
        
    def get_class_distribution(self, data: pd.DataFrame, label_col: str = 'label') -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        if label_col not in data.columns:
            logger.warning(f"Label column '{label_col}' not found in dataset")
            return {}
            
        return data[label_col].value_counts().to_dict()