# wakeword_model.py
import torch
import torch.nn as nn

class WakeWordModel(nn.Module):
    """Wake Word Detection Model wrapping a pretrained conv backbone"""
    def __init__(self, pretrained_model, input_shape=(1, 101, 40), num_classes=2, freeze_conv=True, dropout=0.5):
        super().__init__()
        self.conv1 = pretrained_model.conv1
        self.conv2 = pretrained_model.conv2

        if freeze_conv:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.conv2.parameters():
                param.requires_grad = False

        # Compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = torch.relu(self.conv1(dummy_input))
            x = torch.relu(self.conv2(x))
            flattened_size = x.view(1, -1).shape[1]

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.output(x)
        return x


def load_wakeword_model(model_path, target_length=101, device=None):
    """
    Load the wake word model from checkpoint
    Args:
        model_path: path to .pt checkpoint
        target_length: length of input time dimension
        device: 'cuda' or 'cpu'
    Returns:
        model: loaded PyTorch model on device
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on device: {device}")

    # Base model architecture
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

    pretrained_model = OriginalModel()
    model = WakeWordModel(pretrained_model, input_shape=(1, target_length, 40), num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    return model
