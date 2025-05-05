import pandas as pd
import numpy as np
import logging
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from utils.caching import CACHE_DIR
from typing import List, Generator, Optional, Union

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
            
            info.append({
                "filename": path.name,
                "path": str(path),
                "size_mb": round(size_mb, 2),
                "malicious": is_malicious
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
        if columns is None:
            reader = pd.read_csv(
                file_path,
                header=0,
                chunksize=self.chunk_size,
                low_memory=False,
                # dtype={col: DTYPES.get(col, 'object') for col in DTYPES.keys()}
            )
        else:
            reader = pd.read_csv(
                file_path,
                header=None,
                names=columns,
                chunksize=self.chunk_size,
                low_memory=False,
                # dtype={col: DTYPES.get(col, 'object') for col in columns}
            )
        for chunk in reader:
            if columns is None:
                chunk.columns = [col.strip() for col in chunk.columns]
            
            yield chunk
    
    @staticmethod
    def _process_chunk(chunk: pd.DataFrame, feature_extractor) -> pd.DataFrame:
        """Process a single chunk with feature extraction."""
        return feature_extractor.extract_features(chunk)
    
    def process_file(
        self,
        file_path: Union[str, Path],
        feature_extractor,
        cache_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process an entire file with chunking and optional caching.
        
        Args:
            file_path: Path to the data file
            feature_extractor: FeatureExtractor instance
            cache_key: Key for caching (if None, no caching)
            
        Returns:
            DataFrame with processed features
        """
        file_path = Path(file_path)
        
        # Generate cache key if not provided
        if cache_key is None and self.cache_processed:
            cache_key = f"{file_path.stem}_{feature_extractor.name}"
        
        # Check cache
        cache_path = CACHE_DIR / f"{cache_key}.parquet"
        if self.cache_processed and cache_path.exists():
            logger.info(f"Loading processed data from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Process in chunks
        processed_chunks = []
        for i, chunk in enumerate(self.load_file_chunks(file_path)):
            logger.info(f"Processing chunk {i+1} from {file_path.name}")
            processed = self._process_chunk(chunk, feature_extractor)
            processed_chunks.append(processed)
        
        # Combine processed chunks
        result = pd.concat(processed_chunks, ignore_index=True)

        # Cache result
        if self.cache_processed:
            logger.info(f"Caching processed data: {cache_path}")
            result.to_parquet(cache_path)
        
        return result
    
    def process_files_parallel(
        self,
        file_paths: List[Union[str, Path]],
        feature_extractor,
        sample_fraction: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Process multiple files in parallel.
        
        Args:
            file_paths: List of paths to data files
            feature_extractor: FeatureExtractor instance
            sample_fraction: Fraction of data to sample (None for all)
            
        Returns:
            DataFrame with processed features from all files
        """
        file_paths = [Path(p) for p in file_paths]
        
        logger.info(f"Processing {len(file_paths)} files in parallel")
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_file)(
                file_path=fp,
                feature_extractor=feature_extractor,
                cache_key=f"{fp.stem}_{feature_extractor.name}"
            )
            for fp in file_paths
        )
        
        # Combine results
        combined = pd.concat(results, ignore_index=True)
        
        # Sample if requested
        if sample_fraction is not None and 0 < sample_fraction < 1:
            logger.info(f"Sampling {sample_fraction*100}% of data")
            combined = combined.sample(frac=sample_fraction, random_state=42)
        
        return combined