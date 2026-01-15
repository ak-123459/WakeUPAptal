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
    
    # 3. Load trained classifier weights
    checkpoint = torch.load(pt_model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    # 🔥 DEBUG: Check actual parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model state_dict keys: {len(model.state_dict())}")
    
    # Print layer names to see what's included
    print("\nModel layers:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  - {name}: {num_params:,} parameters")
    
    # 4. UNFREEZE everything for export
    for param in model.parameters():
        param.requires_grad = True
    
    # 5. Evaluation mode
    model.eval()
    
    # 6. Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # 7. Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nTest output shape: {output.shape}")
    
    # 8. Export to ONNX with LEGACY exporter
    import os
    
    print("\n🔧 Exporting with LEGACY exporter (dynamo=False)...")
    
    # 🔥 KEY FIX: Use context manager to disable dynamo
    with torch.no_grad():
        # Force legacy exporter by using dynamo=False
        torch.onnx.export(
            model,
            dummy_input,
            onnx_model_path,
            export_params=True,              # ✅ Must be True
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False,                     # 🔥 Disable dynamo
            verbose=False                     # Set to True for debugging
        )
    
    onnx_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
    print(f"\n✅ Model converted!")
    print(f"   ONNX size: {onnx_size:.2f} MB")
    
    # Expected size check
    expected_size_mb = total_params * 4 / (1024 * 1024)
    print(f"   Expected size: ~{expected_size_mb:.2f} MB")
    
    if onnx_size < expected_size_mb * 0.8:
        print("⚠️  WARNING: ONNX file is smaller than expected!")
    else:
        print("✅ Size looks correct!")
    
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
        input_shape=(1, 101, 40),
        num_classes=2,
        freeze_conv=True,
        dropout=0.5,
        opset_version=11,  # Use 11-14 for better legacy support
    )
