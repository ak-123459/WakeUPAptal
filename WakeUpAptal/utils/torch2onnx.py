import torch
import torch.onnx
from typing import Tuple
import torch.nn as nn



from model import WakeWordModel, load_pretrained_model

def convert_pt_to_onnx(
    pt_model_path: str,
    onnx_model_path: str,
    pretrained_path: str,
    input_shape: tuple = (1, 101, 40),
    num_classes: int = 2,
    freeze_conv: bool = True,
    dropout: float = 0.5,
    opset_version: int = 11
):
    """
    Convert .pt model to ONNX format
    
    Args:
        pt_model_path: Path to your trained .pt model
        onnx_model_path: Output path for ONNX model (e.g., 'model.onnx')
        pretrained_path: Path to pretrained model
        input_shape: Model input shape (channels, height, width)
        num_classes: Number of output classes
        freeze_conv: Whether conv layers were frozen
        dropout: Dropout rate used in model
        opset_version: ONNX opset version (11 or higher recommended)
    """
    
    # 1. Load pretrained model
    pretrained_model = load_pretrained_model(pretrained_path, device='cpu')
    
    # 2. Create model architecture
    model = WakeWordModel(
        pretrained_model=pretrained_model,
        input_shape=input_shape,
        num_classes=num_classes,
        freeze_conv=freeze_conv,
        dropout=dropout
    )
    
    # 3. Load trained weights
    checkpoint = torch.load(pt_model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    # 4. Set to evaluation mode
    model.eval()
    
    # 5. Create dummy input (batch_size=1, channels, height, width)
    dummy_input = torch.randn(1, *input_shape)
    
    # 6. Export to ONNX
    torch.onnx.export(
        model,                          # Model
        dummy_input,                    # Model input
        onnx_model_path,               # Output file
        export_params=True,             # Store trained weights
        opset_version=opset_version,   # ONNX version
        do_constant_folding=True,       # Optimize constants
        input_names=['input'],          # Input name
        output_names=['output'],        # Output name
        dynamic_axes={                  # Variable batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model converted successfully!")
    print(f"   Saved to: {onnx_model_path}")
    
    # 7. Verify the model
    verify_onnx_model(onnx_model_path, dummy_input, model)


def verify_onnx_model(onnx_path: str, dummy_input: torch.Tensor, original_model: nn.Module):
    """Verify ONNX model produces same output as PyTorch model"""
    import onnx
    import onnxruntime as ort
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure is valid")
    
    # Run inference with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = original_model(dummy_input).numpy()
    
    # Get ONNX output
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = abs(pytorch_output - onnx_output).max()
    print(f"✓ Max difference between PyTorch and ONNX: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✅ ONNX model verified successfully!")
    else:
        print("⚠️  Warning: Outputs differ significantly")


# Example usage:
if __name__ == "__main__":
    convert_pt_to_onnx(
        pt_model_path="/content/new_trained_model (1).pt",
        onnx_model_path="hello_aptal.onnx",
        pretrained_path="/content/BOIG/WakeUpAptal/pretrained/pretrained-model.pt",
        input_shape=(1, 101, 40),  # Your input shape
        num_classes=2,
        freeze_conv=True,
        dropout=0.5,
        opset_version=11
    )
