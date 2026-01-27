import argparse
import os
import subprocess
import sys
import yaml
import shutil
import time
from tqdm import tqdm
import re

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
            print(f"📥 Downloading {filename}...")
            subprocess.check_call(["curl", "-L", url, "-o", filepath])
    
    # Also need SALAD weights
    salad_dir = "da3_streaming/weights"
    salad_file = os.path.join(salad_dir, "dino_salad.ckpt")
    if not os.path.exists(salad_file):
        print("🥗 Downloading SALAD weights...")
        if not os.path.exists(salad_dir):
            os.makedirs(salad_dir)
        subprocess.check_call(["curl", "-L", "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt", "-o", salad_file])

def create_config(chunk_size, overlap, loop_enable=True, align_method='sim3', depth_threshold=15.0, loop_chunk_size=10):
    """Creates a temporary config file with specified settings."""
    base_config_path = "da3_streaming/configs/base_config.yaml"
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['Model']['chunk_size'] = chunk_size
    config['Model']['overlap'] = overlap
    config['Model']['loop_enable'] = loop_enable
    config['Model']['align_method'] = align_method
    config['Model']['depth_threshold'] = depth_threshold
    config['Model']['loop_chunk_size'] = loop_chunk_size
    
    output_path = "da3_streaming/configs/temp_config.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    return output_path

def count_images(image_dir):
    """Recursively count images in a directory."""
    count = 0
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Run DA3 Pose Extraction Pipeline")
    parser.add_argument("--image_dir", required=True, help="Path to input images directory")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--chunk_size", type=int, default=20, help="Chunk size for streaming (default: 20)")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap between chunks (default: 10)")
    parser.add_argument("--no_loop", action="store_false", dest="loop_enable", help="Disable loop closure detection")
    parser.add_argument("--align_method", choices=['sim3', 'se3', 'scale+se3'], default='sim3', help="Alignment method (default: sim3)")
    parser.add_argument("--depth_threshold", type=float, default=15.0, help="Depth threshold for alignment (default: 15.0)")
    parser.add_argument("--loop_chunk_size", type=int, default=10, help="Chunk size for loop alignment (default: 10). Reduce if OOM happens at the end.")
    parser.set_defaults(loop_enable=True)
    
    args = parser.parse_args()
    
    start_time = time.time()
    num_images = count_images(args.image_dir)
    
    print("\n" + "="*50)
    print("🚀 DA3 POSE EXTRACTION PIPELINE STARTING")
    print("="*50)
    print(f"📂 Input Images:  {num_images} images in {args.image_dir}")
    print(f"📦 Segments:      Chunk Size={args.chunk_size}, Overlap={args.overlap}")
    print(f"� Loop Settings: Enabled={'✅' if args.loop_enable else '❌'}, Loop Chunk Size={args.loop_chunk_size}")
    print(f"🛠️  Alignment:     Method={args.align_method}")
    print(f"📐 Threshold:     {args.depth_threshold}m")
    print("="*50 + "\n")

    print("🛰️  Step 1: Checking Weights...")
    download_weights()
    
    print(f"⚙️  Step 2: Creating Config...")
    config_path = create_config(args.chunk_size, args.overlap, args.loop_enable, args.align_method, args.depth_threshold, args.loop_chunk_size)
    
    print(f"🔥 Step 3: Running Streaming Pipeline...")
    
    # Environment Setup
    env = os.environ.copy()
    env["PYTHONPATH"] = "../src:."
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    # Command Construction
    rel_image_dir = os.path.abspath(args.image_dir)
    rel_output_dir = os.path.abspath(args.output_dir)
    rel_config_path = f"configs/{os.path.basename(config_path)}"
    
    streaming_cmd = [
        "python", "-u", "da3_streaming.py", # -u for unbuffered output
        "--image_dir", rel_image_dir,
        "--output_dir", rel_output_dir,
        "--config", rel_config_path
    ]
    
    pbar = None
    process = subprocess.Popen(
        streaming_cmd, 
        cwd="da3_streaming", 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )
    
    progress_pattern = re.compile(r"\[Progress\]: (\d+)/(\d+)")
    
    for line in iter(process.stdout.readline, ''):
        match = progress_pattern.search(line)
        if match:
            curr, total = map(int, match.groups())
            if pbar is None:
                pbar = tqdm(total=total, desc="📽️  Extracting Poses", unit="chunk")
            pbar.update(curr - pbar.n)
        elif "DA3-Streaming done" in line:
            if pbar: pbar.update(pbar.total - pbar.n)
        # else:
        #    print(line.strip()) # Uncomment for full debug logs
    
    process.stdout.close()
    return_code = process.wait()
    if pbar: pbar.close()
    
    if return_code != 0:
        print(f"❌ Error: Streaming pipeline failed with code {return_code}")
        sys.exit(return_code)

    print("\n📦 Step 4: Exporting to GLB...")
    ply_path = os.path.join(args.output_dir, "camera_poses.ply")
    glb_path = os.path.join(args.output_dir, "camera_poses.glb")
    
    if os.path.exists(ply_path):
        subprocess.check_call(["python", "scripts/convert_poses_to_glb.py", ply_path, glb_path])
    else:
        print("❌ Error: camera_poses.ply not found.")
        sys.exit(1)
        
    pcd_path = os.path.join(args.output_dir, "pcd", "combined_pcd.ply")
    pcd_glb_path = os.path.join(args.output_dir, "scene.glb")
    if os.path.exists(pcd_path):
        print(f"✨ Converting scene to GLB...")
        subprocess.check_call(["python", "scripts/convert_poses_to_glb.py", pcd_path, pcd_glb_path])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*50)
    print("✅ POSE EXTRACTION COMPLETE!")
    print("="*50)
    print(f"⏱️  Duration:     {duration/60:.1f} minutes")
    print(f"📂 Output Dir:   {args.output_dir}")
    print(f"🎮 Visualization: Open 'camera_poses.glb' or 'scene.glb' in any 3D viewer")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
