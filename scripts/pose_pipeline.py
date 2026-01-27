import argparse
import os
import subprocess
import sys
import yaml
import shutil

def download_weights():
    """Downloads DA3-LARGE weights if they don't exist."""
    weights_dir = "da3_streaming/weights_large"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    files = {
        "config.json": "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/config.json",
        "model.safetensors": "https://huggingface.co/depth-anything/DA3-LARGE/resolve/main/model.safetensors"
    }
    
    for filename, url in files.items():
        filepath = os.path.join(weights_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            subprocess.check_call(["curl", "-L", url, "-o", filepath])
    
    # Also need SALAD weights
    salad_dir = "da3_streaming/weights"
    salad_file = os.path.join(salad_dir, "dino_salad.ckpt")
    if not os.path.exists(salad_file):
        print("Downloading SALAD weights...")
        if not os.path.exists(salad_dir):
            os.makedirs(salad_dir)
        subprocess.check_call(["curl", "-L", "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt", "-o", salad_file])

def create_config(chunk_size, overlap, loop_enable=True, align_method='sim3', depth_threshold=15.0):
    """Creates a temporary config file with specified settings."""
    # Load base config
    base_config_path = "da3_streaming/configs/base_config.yaml"
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update values
    config['Model']['chunk_size'] = chunk_size
    config['Model']['overlap'] = overlap
    config['Model']['loop_enable'] = loop_enable
    config['Model']['align_method'] = align_method
    config['Model']['depth_threshold'] = depth_threshold
    
    output_path = "da3_streaming/configs/temp_config.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Run DA3 Pose Extraction Pipeline")
    parser.add_argument("--image_dir", required=True, help="Path to input images directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--chunk_size", type=int, default=20, help="Chunk size for streaming (default: 20)")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap between chunks (default: 10)")
    parser.add_argument("--no_loop", action="store_false", dest="loop_enable", help="Disable loop closure detection")
    parser.add_argument("--align_method", choices=['sim3', 'se3', 'scale+se3'], default='sim3', help="Alignment method (default: sim3)")
    parser.add_argument("--depth_threshold", type=float, default=15.0, help="Depth threshold for alignment (default: 15.0)")
    parser.set_defaults(loop_enable=True)
    
    args = parser.parse_args()
    
    print("Step 1: Checking Weights...")
    download_weights()
    
    print(f"Step 2: Creating Config (Chunk: {args.chunk_size}, Overlap: {args.overlap}, Loop: {args.loop_enable})...")
    config_path = create_config(args.chunk_size, args.overlap, args.loop_enable, args.align_method, args.depth_threshold)
    
    print("Step 3: Running Streaming pipeline...")
    # adapting command:
    # cd da3_streaming && PYTHONPATH=../src:. PYTORCH_ALLOC_CONF=expandable_segments:True python da3_streaming.py ...
    
    cmd = [
        "python", "da3_streaming/da3_streaming.py",
        "--image_dir", args.image_dir,
        "--output_dir", args.output_dir,
        "--config", config_path
    ]
    
    # We need to set up environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:." # We are running from root, so src is local. da3_streaming imports loop_utils (local) and depth_anything_3 (src).
    # actually, da3_streaming.py expects to be run from inside da3_streaming dir usually?
    # or if we run from root, we need to make sure loop_utils is importable.
    # `da3_streaming.py` does `from loop_utils...`. `loop_utils` is in `da3_streaming/loop_utils`.
    # So `da3_streaming` package is essentially `da3_streaming/`.
    # If we run from root, `import loop_utils` will fail unless `da3_streaming` is in path? 
    # The pixi task `da3-streaming` does `cd da3_streaming`.
    # Let's mimic that behavior to be safe.
    
    # Adjust paths for CWD=da3_streaming
    rel_image_dir = os.path.abspath(args.image_dir)
    rel_output_dir = os.path.abspath(args.output_dir)
    rel_config_path = os.path.basename(config_path) # since it's in configs/, relative to da3_streaming it is configs/temp_config.yaml
    rel_config_path = f"configs/{os.path.basename(config_path)}"
    
    streaming_cmd = [
        "python", "da3_streaming.py",
        "--image_dir", rel_image_dir,
        "--output_dir", rel_output_dir,
        "--config", rel_config_path
    ]
    
    env["PYTHONPATH"] = "../src:."
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    print(f"Executing: {' '.join(streaming_cmd)}")
    subprocess.check_call(streaming_cmd, cwd="da3_streaming", env=env)
    
    print("Step 4: Converting to GLB...")
    # Check if we have ply
    ply_path = os.path.join(args.output_dir, "camera_poses.ply")
    glb_path = os.path.join(args.output_dir, "camera_poses.glb")
    
    if os.path.exists(ply_path):
        convert_cmd = [
            "python", "scripts/convert_poses_to_glb.py",
            ply_path,
            glb_path
        ]
        subprocess.check_call(convert_cmd) # run from root
    else:
        print("Error: camera_poses.ply not found. Streaming might have failed.")
        sys.exit(1)
        
    # Convert scene point cloud
    pcd_path = os.path.join(args.output_dir, "pcd", "combined_pcd.ply")
    pcd_glb_path = os.path.join(args.output_dir, "scene.glb")
    
    if os.path.exists(pcd_path):
        print(f"Converting scene point cloud to {pcd_glb_path}...")
        convert_cmd = [
            "python", "scripts/convert_poses_to_glb.py",
            pcd_path,
            pcd_glb_path
        ]
        subprocess.check_call(convert_cmd)
    
    print(f"Success! Outputs saved to:\n- {glb_path}\n- {pcd_glb_path} (if scene exists)")

if __name__ == "__main__":
    main()
