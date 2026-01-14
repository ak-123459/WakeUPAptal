
# ============================================================================
# FILE 2: data_preprocessing.py
# ============================================================================
"""
Data loading, preprocessing, and feature extraction
Usage: python data_preprocessing.py --config config.json
"""

import torch
import numpy as np
import os
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple, Optional
import logging

from utils.manage_audio import AudioPreprocessor
from config import Config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Handles feature extraction and caching"""
    
    def __init__(self, audio_config, cache_dir: str = 'mfcc_cache'):
        self.audio_config = audio_config
        self.cache_dir = cache_dir
        self.preprocessor = AudioPreprocessor(
            sr=audio_config.sample_rate,
            n_dct_filters=audio_config.n_dct_filters,
            n_mels=audio_config.n_mels,
            f_max=audio_config.f_max,
            f_min=audio_config.f_min,
            n_fft=audio_config.n_fft,
            hop_ms=audio_config.hop_ms
        )
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, filepath: str, split: str) -> str:
        """Generate cache file path"""
        file_hash = abs(hash(filepath))
        cache_subdir = os.path.join(self.cache_dir, split)
        os.makedirs(cache_subdir, exist_ok=True)
        return os.path.join(cache_subdir, f"{file_hash}.npy")
    
    def _process_single_file(self, filepath: str, split: str) -> np.ndarray:
        """Process single audio file and cache result"""
        cache_path = self._get_cache_path(filepath, split)
        
        # Check cache
        if os.path.exists(cache_path):
            return cache_path
        
        # Compute MFCC
        mfccs = self.preprocessor.compute_mfccs(filepath)
        audio_tensor = torch.FloatTensor(mfccs)
        
        # Pad or truncate to target length
        target_length = self.audio_config.target_length
        current_length = audio_tensor.shape[0]
        
        if current_length < target_length:
            padding = torch.zeros(
                target_length - current_length,
                audio_tensor.shape[1],
                audio_tensor.shape[2]
            )
            audio_tensor = torch.cat([audio_tensor, padding], dim=0)
        elif current_length > target_length:
            audio_tensor = audio_tensor[:target_length]
        
        # Permute to [channels, time, mels]
        audio_tensor = audio_tensor.permute(2, 0, 1)
        
        # Save to cache
        np.save(cache_path, audio_tensor.numpy())
        return cache_path
    
    def process_batch(self, file_list: List[str], split: str = 'train') -> List[str]:
        """Process batch of files and return cache paths"""
        logger.info(f"Processing {len(file_list)} files for {split} set...")
        cached_paths = []
        
        for filepath in tqdm(file_list, desc=f"Extracting features ({split})"):
            cache_path = self._process_single_file(filepath, split)
            cached_paths.append(cache_path)
        
        logger.info(f"✓ Processed {len(cached_paths)} files")
        return cached_paths
    
    def clear_cache(self):
        """Clear all cached features"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            logger.info(f"✓ Cleared cache directory: {self.cache_dir}")


class DataLoader:
    """Handles data loading and splitting"""
    
    def __init__(self, data_config):
        self.config = data_config
        np.random.seed(self.config.random_seed)
    
    def load_files_and_labels(self) -> Tuple[List[str], List[int]]:
        """Load file paths and labels from directories"""
        logger.info("Loading dataset...")
        
        # Load positive samples
        positive_files = glob.glob(os.path.join(self.config.positive_dir, "*.wav"))
        if self.config.max_samples_per_class and len(positive_files) > self.config.max_samples_per_class:
            positive_files = list(np.random.choice(
                positive_files,
                self.config.max_samples_per_class,
                replace=False
            ))
        positive_labels = [1] * len(positive_files)
        
        # Load negative samples
        negative_files = glob.glob(os.path.join(self.config.negative_dir, "*.wav"))
        if self.config.max_samples_per_class and len(negative_files) > self.config.max_samples_per_class:
            negative_files = list(np.random.choice(
                negative_files,
                self.config.max_samples_per_class,
                replace=False
            ))
        negative_labels = [0] * len(negative_files)
        
        # Combine
        all_files = positive_files + negative_files
        all_labels = positive_labels + negative_labels
        
        logger.info(f"  Positive samples: {len(positive_files)}")
        logger.info(f"  Negative samples: {len(negative_files)}")
        logger.info(f"  Total samples: {len(all_files)}")
        
        return all_files, all_labels
    
    def split_data(self, files: List[str], labels: List[int]) -> Dict[str, Tuple[List[str], List[int]]]:
        """Split data into train/val/test sets"""
        logger.info("Splitting dataset...")
        
        # Calculate split sizes
        test_size = self.config.test_ratio
        val_size = self.config.val_ratio / (1 - self.config.test_ratio)
        
        # First split: separate test set
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            files, labels,
            test_size=test_size,
            random_state=self.config.random_seed,
            stratify=labels
        )
        
        # Second split: separate train and validation
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=val_size,
            random_state=self.config.random_seed,
            stratify=train_val_labels
        )
        
        logger.info(f"  Train: {len(train_files)} samples")
        logger.info(f"  Val:   {len(val_files)} samples")
        logger.info(f"  Test:  {len(test_files)} samples")
        
        return {
            'train': (train_files, train_labels),
            'val': (val_files, val_labels),
            'test': (test_files, test_labels)
        }


class WakeWordDataset(Dataset):
    """PyTorch Dataset for wake word detection"""
    
    def __init__(
        self,
        cached_files: List[str],
        labels: List[int],
        augment: bool = False,
        augmentation_config: Optional[DataConfig] = None
    ):
        self.cached_files = cached_files
        self.labels = labels
        self.augment = augment
        self.aug_config = augmentation_config
    
    def __len__(self) -> int:
        return len(self.cached_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load cached feature
        audio_tensor = torch.FloatTensor(np.load(self.cached_files[idx]))
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augment and self.aug_config:
            audio_tensor = self._augment(audio_tensor)
        
        return audio_tensor, label
    
    def _augment(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        aug_prob = self.aug_config.augmentation_prob
        
        # Additive noise
        if np.random.random() < aug_prob:
            noise = torch.randn_like(audio) * self.aug_config.noise_std
            audio = audio + noise
        
        # Time shift
        if np.random.random() < aug_prob:
            shift = np.random.randint(
                -self.aug_config.shift_range,
                self.aug_config.shift_range
            )
            audio = torch.roll(audio, shift, dims=1)
        
        # Amplitude scaling
        if np.random.random() < aug_prob:
            scale = np.random.uniform(*self.aug_config.scale_range)
            audio = audio * scale
        
        return audio


class DataPipeline:
    """Complete data preprocessing pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config.data)
        self.feature_extractor = FeatureExtractor(
            config.audio,
            cache_dir=config.data.cache_dir
        )
    
    def prepare_datasets(self, force_recompute: bool = False) -> Dict[str, WakeWordDataset]:
        """Prepare train/val/test datasets"""
        logger.info("="*80)
        logger.info("DATA PREPROCESSING PIPELINE")
        logger.info("="*80)
        
        # Clear cache if requested
        if force_recompute:
            self.feature_extractor.clear_cache()
        
        # Load and split data
        files, labels = self.data_loader.load_files_and_labels()
        splits = self.data_loader.split_data(files, labels)
        
        # Extract features for each split
        datasets = {}
        for split_name, (split_files, split_labels) in splits.items():
            cached_files = self.feature_extractor.process_batch(split_files, split_name)
            
            # Create dataset with augmentation only for training
            augment = (split_name == 'train') and self.config.data.augment_train
            datasets[split_name] = WakeWordDataset(
                cached_files,
                split_labels,
                augment=augment,
                augmentation_config=self.config.data if augment else None
            )
        
        logger.info("="*80)
        logger.info("✓ Data preprocessing complete")
        logger.info("="*80)
        
        return datasets
    
    def get_dataloaders(
        self,
        datasets: Dict[str, WakeWordDataset]
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """Create PyTorch DataLoaders"""
        batch_size = self.config.training.batch_size
        num_workers = self.config.training.num_workers
        pin_memory = self.config.training.pin_memory
        
        dataloaders = {
            'train': torch.utils.data.DataLoader(
                datasets['train'],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            ),
            'val': torch.utils.data.DataLoader(
                datasets['val'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            ),
            'test': torch.utils.data.DataLoader(
                datasets['test'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        }
        
        return dataloaders

