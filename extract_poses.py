import glob
import os
import torch
import numpy as np
import sys

# Add src to path manually if needed
sys.path.append('src')

from depth_anything_3.api import DepthAnything3

def main():
    print("Starting script...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Model Name
    model_name = "depth-anything/DA3-SMALL"
    
    print(f"Loading model: {model_name}")
    # Initialize model
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    
    # Image Directory
    image_dir = "/workspace/datasets/voiture_dure/images"
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*"))) # Assuming images are directly in this folder
    
    # Filter for image extensions if necessary, but glob "*" is usually fine if dir is clean
    # Better to be safe
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    
    # Run Inference in batches
    print("Running inference...")
    batch_size = 5 # Small batch size to be safe
    all_extrinsics = []
    all_intrinsics = []
    
    import math
    num_batches = math.ceil(len(image_paths) / batch_size)
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_paths = image_paths[i*batch_size : (i+1)*batch_size]
            print(f"Processing batch {i+1}/{num_batches}...")
            
            # Run inference on batch
            prediction = model.inference(batch_paths)
            
            # Append results
            # prediction.extrinsics is [B, 3, 4] or similar
            if prediction.extrinsics is not None:
                all_extrinsics.append(prediction.extrinsics)
            if prediction.intrinsics is not None:
                all_intrinsics.append(prediction.intrinsics)
                
            # Clear cache
            torch.cuda.empty_cache()

    # Concatenate results
    if all_extrinsics:
        extrinsics = np.concatenate(all_extrinsics, axis=0)
        intrinsics = np.concatenate(all_intrinsics, axis=0)
    else:
        print("No results extracted.")
        return
    
    print(f"Extrinsics shape: {extrinsics.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")
    
    # Save to NPZ
    output_path = "poses.npz"
    print(f"Saving poses to {output_path}")
    np.savez(output_path, extrinsics=extrinsics, intrinsics=intrinsics, image_paths=image_paths)
    print("Done.")

if __name__ == "__main__":
    main()
