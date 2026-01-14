import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import wave
from pathlib import Path
from sklearn.model_selection import train_test_split
import librosa
from scipy.fftpack import dct

class AudioPreprocessor:
    def __init__(
        self,
        sr=16000,
        n_dct_filters=40,
        n_mels=40,
        f_max=4000,
        f_min=20,
        n_fft=480,
        hop_ms=10
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_dct_filters = n_dct_filters

        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = int(sr * hop_ms / 1000)

        # Create DCT-II filter bank (MFCC step)
        self.dct_filters = self._create_dct_filters(
            n_dct_filters, n_mels
        )

    def _create_dct_filters(self, n_filters, n_input):
        """
        Create DCT-II filter bank manually
        Shape: (n_dct_filters, n_mels)
        """
        basis = np.empty((n_filters, n_input), dtype=np.float32)

        # First coefficient (energy)
        basis[0, :] = 1.0 / np.sqrt(n_input)

        samples = np.arange(1, 2 * n_input, 2) * np.pi / (2.0 * n_input)

        for i in range(1, n_filters):
            basis[i, :] = (
                np.cos(i * samples) * np.sqrt(2.0 / n_input)
            )

        return basis

    def compute_mfccs(self, audio_path):
        """Load audio file and compute MFCCs"""
        # Load audio file
        if isinstance(audio_path, str):
            with wave.open(audio_path, 'rb') as wf:
                n_frames = wf.getnframes()
                audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                audio_data = audio_data.astype(float) / np.iinfo(np.int16).max
        else:
            audio_data = audio_path
        
        # Compute mel spectrogram
        data = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        
        # Log compression
        data[data > 0] = np.log(data[data > 0])
        
        # Apply DCT filters
        data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        
        return data


class WakeWordDataset(Dataset):
    """Dataset for wake word detection with positive and negative samples"""
    
    def __init__(self, file_list, labels, preprocessor, augment=True):
        """
        Args:
            file_list: List of audio file paths
            labels: List of corresponding labels (0 or 1)
            preprocessor: AudioPreprocessor instance
            augment: Whether to apply data augmentation
        """
        self.file_list = file_list
        self.labels = labels
        self.preprocessor = preprocessor
        self.augment = augment
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        label = self.labels[idx]
        
        # Process audio using your preprocessor
        mfccs = self.preprocessor.compute_mfccs(filepath)
        
        # Convert to tensor and add channel dimension
        audio_tensor = torch.FloatTensor(mfccs)
        
        # Data augmentation for training (helps prevent overfitting)
        if self.augment:
            # 1. Add small random noise
            if np.random.random() < 0.3:
                noise = torch.randn_like(audio_tensor) * 0.005
                audio_tensor = audio_tensor + noise
            
            # 2. Time shifting (small shift)
            if np.random.random() < 0.3:
                shift = np.random.randint(-5, 5)
                audio_tensor = torch.roll(audio_tensor, shift, dims=0)
            
            # 3. Amplitude scaling
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.8, 1.2)
                audio_tensor = audio_tensor * scale
        
        return audio_tensor, label


class WakeWordModel(nn.Module):
    def __init__(self, pretrained_model, num_classes=2, freeze_conv=True, dropout=0.5):
        super().__init__()
        # Copy pretrained conv layers
        self.conv1 = pretrained_model.conv1
        self.conv2 = pretrained_model.conv2
        
        # Freeze conv layers to preserve pretrained knowledge
        if freeze_conv:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.conv2.parameters():
                param.requires_grad = False
        
        # Add dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
        # NEW output layer for binary classification
        self.output = nn.Linear(26624, num_classes)
    
    def forward(self, x):
        # Use pretrained feature extraction
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        # Apply dropout before final layer
        x = self.dropout(x)
        x = self.output(x)
        return x


def load_and_split_data(positive_dir, negative_dir, val_split=0.2, test_split=0.1):
    """
    Load data from your existing folder structure and split into train/val/test
    
    Args:
        positive_dir: Path to 'process_dataset/positive' folder
        negative_dir: Path to 'process_dataset/negative' folder
        val_split: Fraction for validation set
        test_split: Fraction for test set
    """
    print("Loading data from folders...")
    
    # Load positive samples
    positive_files = glob.glob(os.path.join(positive_dir, "*.wav"))
    positive_labels = [1] * len(positive_files)
    
    # Load negative samples
    negative_files = glob.glob(os.path.join(negative_dir, "*.wav"))
    negative_labels = [0] * len(negative_files)
    
    # Combine all data
    all_files = positive_files + negative_files
    all_labels = positive_labels + negative_labels
    
    print(f"  Positive samples: {len(positive_files)}")
    print(f"  Negative samples: {len(negative_files)}")
    print(f"  Total samples: {len(all_files)}")
    
    # First split: separate test set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=test_split, random_state=42, stratify=all_labels
    )
    
    # Second split: separate train and validation
    val_ratio = val_split / (1 - test_split)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, test_size=val_ratio, random_state=42, stratify=train_val_labels
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val:   {len(val_files)} samples")
    print(f"  Test:  {len(test_files)} samples")
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For computing precision/recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate metrics
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    # Calculate precision and recall
    precision = 100. * true_positives / (true_positives + false_positives + 1e-8)
    recall = 100. * true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return val_loss, val_acc, precision, recall, f1


def main():
    # ===== CONFIGURATION =====
    POSITIVE_DIR = "process_dataset/positive"  # Your positive samples folder
    NEGATIVE_DIR = "process_dataset/negative"  # Your negative samples folder
    PRETRAINED_MODEL_PATH = "/content/google-speech-dataset.pt"
    
    # Training parameters (adjusted for small dataset to prevent overfitting)
    BATCH_SIZE = 16  # Smaller batch size for small datasets
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-4  # Lower learning rate
    WEIGHT_DECAY = 1e-4  # L2 regularization to prevent overfitting
    DROPOUT = 0.5  # Dropout rate
    
    VAL_SPLIT = 0.2  # 20% for validation
    TEST_SPLIT = 0.1  # 10% for final testing
    
    # Early stopping parameters
    PATIENCE = 15  # Stop if no improvement for 15 epochs
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ===== LOAD DATA =====
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = \
        load_and_split_data(POSITIVE_DIR, NEGATIVE_DIR, VAL_SPLIT, TEST_SPLIT)
    
    # ===== INITIALIZE AUDIO PREPROCESSOR =====
    preprocessor = AudioPreprocessor(
        sr=16000,
        n_dct_filters=40,
        n_mels=40,
        f_max=4000,
        f_min=20,
        n_fft=480,
        hop_ms=10
    )
    
    # ===== CREATE DATASETS =====
    train_dataset = WakeWordDataset(train_files, train_labels, preprocessor, augment=True)
    val_dataset = WakeWordDataset(val_files, val_labels, preprocessor, augment=False)
    test_dataset = WakeWordDataset(test_files, test_labels, preprocessor, augment=False)
    
    # ===== CREATE DATALOADERS =====
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ===== LOAD PRETRAINED MODEL =====
    class OriginalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(20, 8))
            self.conv2 = nn.Conv2d(64, 64, kernel_size=(10, 4))
            self.output = nn.Linear(26624, 12)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.output(x)
            return x
    
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu')
    pretrained_model = OriginalModel()
    pretrained_model.load_state_dict(checkpoint)
    
    # ===== CREATE WAKE WORD MODEL =====
    model = WakeWordModel(pretrained_model, num_classes=2, freeze_conv=True, dropout=DROPOUT)
    model = model.to(device)
    
    print("\nModel architecture:")
    print(f"  Conv layers: FROZEN (using pretrained knowledge)")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Output layer: TRAINABLE (2 classes)")
    
    # ===== LOSS AND OPTIMIZER =====
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize the output layer + use weight decay for regularization
    optimizer = torch.optim.Adam(
        model.output.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # ===== TRAINING LOOP =====
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*80)
    print("Starting Training (with overfitting prevention)")
    print("="*80)
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion, device)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Precision:  {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_wake_word_model.pt')
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping triggered! No improvement for {PATIENCE} epochs.")
            break
    
    # ===== FINAL EVALUATION ON TEST SET =====
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_wake_word_model.pt'))
    
    test_loss, test_acc, test_precision, test_recall, test_f1 = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Test Loss:      {test_loss:.4f}")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Precision:      {test_precision:.2f}%")
    print(f"  Recall:         {test_recall:.2f}%")
    print(f"  F1 Score:       {test_f1:.2f}%")
    print("\n" + "="*80)
    print(f"Training Complete! Model saved as 'best_wake_word_model.pt'")
    print("="*80)


if __name__ == "__main__":
    main()