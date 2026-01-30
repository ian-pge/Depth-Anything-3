import argparse
import os
import subprocess
import sys
import yaml
import shutil
import time
from tqdm import tqdm
import re
import numpy as np
import trimesh
import matplotlib.cm as cm

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
    parser.add_argument("--output_dir", required=False, default=None, help="Path to output directory (default: <input_dir>/da3)")
    parser.add_argument("--chunk_size", type=int, default=20, help="Chunk size for streaming (default: 20)")
    parser.add_argument("--overlap", type=int, default=10, help="Overlap between chunks (default: 10)")
    parser.add_argument("--no_loop", action="store_false", dest="loop_enable", help="Disable loop closure detection")
    parser.add_argument("--align_method", choices=['sim3', 'se3', 'scale+se3'], default='sim3', help="Alignment method (default: sim3)")
    parser.add_argument("--depth_threshold", type=float, default=15.0, help="Depth threshold for alignment (default: 15.0)")
    parser.add_argument("--loop_chunk_size", type=int, default=10, help="Chunk size for loop alignment (default: 10). Reduce if OOM happens at the end.")
    parser.set_defaults(loop_enable=True)
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Smart Path Handling
    raw_image_dir = args.image_dir
    
    # Check if 'images' subdirectory exists
    potential_sub = os.path.join(raw_image_dir, "images")
    if os.path.isdir(potential_sub):
        print(f"📂 Auto-detected 'images' subdirectory. Using {potential_sub} as input.")
        actual_input_dir = potential_sub
        dataset_root = raw_image_dir
    else:
        actual_input_dir = raw_image_dir
        dataset_root = raw_image_dir

    # Default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(dataset_root, "da3")
        print(f"📂 No output directory specified. Defaulting to {args.output_dir}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_images = count_images(actual_input_dir)
    
    print("\n" + "="*50)
    print("🚀 DA3 POSE EXTRACTION PIPELINE STARTING")
    print("="*50)
    print(f"📂 Input Images:  {num_images} images in {actual_input_dir}")
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
    
    # Create a wrapper script to patch DA3_Streaming -> REMOVED
    # We now use the patched da3_streaming.py directly
    
    # Command Construction
    rel_image_dir = os.path.abspath(actual_input_dir)
    rel_output_dir = os.path.abspath(args.output_dir)
    rel_config_path = f"configs/{os.path.basename(config_path)}"
    
    streaming_cmd = [
        "python", "-u", "da3_streaming.py", # Run original script directly
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
    loop_list_pattern = re.compile(r"^\[(\(\d+, \(\d+, \d+\), \d+, \(\d+, \d+\)\)(, )?)+\]$")
    loop_item_pattern = re.compile(r"^\(\d+, \(\d+, \d+\), \d+, \(\d+, \d+\)\)$")
    apply_pattern = re.compile(r"Applying (\d+) -> (\d+) \(Total (\d+)\)")
    
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        
        # 1. Main Extraction Progress
        match = progress_pattern.search(line)
        if match:
            curr, total = map(int, match.groups())
            if pbar is None:
                pbar = tqdm(total=total, desc="📽️  Extracting Poses ", unit="chunk")
            pbar.update(curr - pbar.n)
            continue

        # 2. Loop Refinement Progress
        if loop_list_pattern.match(line):
            # Detected the list of loop pairs, initialize loop pbar
            # The line looks like [(31, (1574, 1594), 2, (129, 149)), ...]
            num_loops = line.count("), ") + 1 if line != "[]" else 0
            if num_loops > 0:
                if pbar: pbar.close()
                pbar = tqdm(total=num_loops, desc="🔄  Refining Loops   ", unit="loop")
            continue
        
        if loop_item_pattern.match(line):
            if pbar and pbar.desc.startswith("🔄"):
                pbar.update(1)
            continue

        # 3. Applying Alignment Progress
        match = apply_pattern.search(line)
        if match:
            curr, _, total = map(int, match.groups())
            if pbar is None or not pbar.desc.startswith("🔗"):
                if pbar: pbar.close()
                pbar = tqdm(total=int(total), desc="🔗  Applying Aligned ", unit="chunk")
            pbar.update(curr - pbar.n)
            continue

        if "DA3-Streaming done" in line:
            if pbar: pbar.update(pbar.total - pbar.n)
        
        print(line) # Debug: uncomment to see raw logs if something is stuck
    
    process.stdout.close()
    return_code = process.wait()
    if pbar: pbar.close()
    
    if return_code != 0:
        print(f"❌ Error: Streaming pipeline failed with code {return_code}")
        sys.exit(return_code)

    print("\n📦 Step 4: Exporting to GLB...")
    try:
        generate_glb_from_artifacts(args.output_dir)
    except Exception as e:
        print(f"❌ Error exporting GLB: {e}")
        import traceback
        traceback.print_exc()

    end_time = time.time()
    duration = end_time - start_time
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "="*50)
    print(f"✅ PIPELINE COMPLETED in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"📂 Output saved to: {args.output_dir}")
    print("="*50)

# =========================================================================
# GLB Export Logic
# =========================================================================

def _camera_frustum_lines(K, w2c, W, H, scale):
    corners = np.array([[0, 0, 1], [W-1, 0, 1], [W-1, H-1, 1], [0, H-1, 1]], dtype=float)
    K_inv = np.linalg.inv(K)
    c2w = np.linalg.inv(w2c)
    # Check shape
    if c2w.shape == (3, 4):
         c2w = np.vstack([c2w, [0,0,0,1]])
    
    Cw = (c2w @ np.array([0,0,0,1]))[:3]
    rays = (K_inv @ corners.T).T
    plane_cam = rays * scale
    
    plane_w = []
    for p in plane_cam:
        pw = (c2w @ np.array([p[0], p[1], p[2], 1]))[:3]
        plane_w.append(pw)
    plane_w = np.stack(plane_w, 0)
    
    segs = []
    for k in range(4): segs.append(np.stack([Cw, plane_w[k]]))
    order = [0, 1, 2, 3, 0]
    for a, b in zip(order[:-1], order[1:]): segs.append(np.stack([plane_w[a], plane_w[b]]))
    return np.stack(segs)

def _index_color_rgb(i, n):
    h = (i + 0.5) / max(n, 1)
    # Use simple HSV to RGB conversion or matplotlib
    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb([h, 0.85, 0.95])
    return (np.array(rgb) * 255).astype(np.uint8)

def generate_glb_from_artifacts(output_dir):
    poses_path = os.path.join(output_dir, "camera_poses.txt")
    intrinsics_path = os.path.join(output_dir, "intrinsic.txt")
    pcd_path = os.path.join(output_dir, "pcd", "combined_pcd.ply")
    
    if not os.path.exists(poses_path):
        print(f"⚠️  Poses file not found at {poses_path}. Skipping GLB export.")
        return
        
    print(f"Reading artifacts from {output_dir}...")
    
    # 1. Read Poses
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            mat = np.array(list(map(float, line.strip().split()))).reshape(4, 4)
            poses.append(mat)
    
    # 2. Read Intrinsics
    intrinsics = []
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                # Format: fx fy cx cy
                K = np.eye(3)
                K[0,0] = vals[0]
                K[1,1] = vals[1]
                K[0,2] = vals[2]
                K[1,2] = vals[3]
                intrinsics.append(K)
    else:
        print("⚠️  Intrinsics file not found. Using identity.")
        intrinsics = [np.eye(3)] * len(poses)
        
    # 3. Read Point Cloud (Optional)
    points = None
    colors = None
    if os.path.exists(pcd_path):
        try:
            print(f"Loading Point Cloud from {pcd_path}...")
            pcd = trimesh.load(pcd_path)
            if hasattr(pcd, 'vertices'):
                 points = pcd.vertices
                 colors = pcd.colors if hasattr(pcd, 'colors') else None
        except Exception as e:
            print(f"⚠️  Failed to load point cloud: {e}")
            
    # 4. Generate Scene
    scene = trimesh.Scene()
    
    # Add Point Cloud
    if points is not None and len(points) > 0:
        pc = trimesh.points.PointCloud(vertices=points, colors=colors)
        scene.add_geometry(pc)
        
    # Determine scale for cameras
    if points is not None and len(points) > 1:
        lo = np.percentile(points, 5, axis=0)
        hi = np.percentile(points, 95, axis=0)
        diag = np.linalg.norm(hi - lo)
        scene_scale = float(diag) if diag > 0 else 1.0
    else:
        scene_scale = 1.0
    
    camera_size = 0.03 * scene_scale
    
    # Add Cameras
    # We create one path per camera to correctly handle colors per entity.
    # While less efficient for thousands of objects, it ensures trimesh maps colors correctly.
    
    pose_paths = []
    N = len(poses)

    for i in range(N):
        c2w = poses[i]
        w2c = np.linalg.inv(c2w)
        
        K = intrinsics[i]
        H = int(K[1, 2] * 2)
        W = int(K[0, 2] * 2)
        
        lines = _camera_frustum_lines(K, w2c, W, H, camera_size) # (8, 2, 3)
        
        # Create a Path3D for this camera
        # trimesh.load_path(lines) creates a Path3D with entities
        cam_path = trimesh.load_path(lines)
        
        # Assign color to all entities in this path
        color_rgb = _index_color_rgb(i, N) # (3,) uint8
        # Path3D.colors expects (N_entities, 4) usually, or (N_entities, 3)
        # We append alpha=255
        color_rgba = np.append(color_rgb, 255)
        
        # Tile for all entities in this path
        # Note: trimesh.load_path([8 segments]) usually creates 1 or more entities.
        # We check len(cam_path.entities)
        cam_path.colors = np.tile(color_rgba, (len(cam_path.entities), 1))
        
        scene.add_geometry(cam_path)
        pose_paths.append(cam_path)
        
    out_glb = os.path.join(output_dir, "scene.glb")
    scene.export(out_glb)
    print(f"✨ Exported scene GLB to {out_glb}")
    
    # Also export just poses
    scene_poses = trimesh.Scene()
    for p in pose_paths:
        scene_poses.add_geometry(p)
    out_poses = os.path.join(output_dir, "camera_poses.glb")
    scene_poses.export(out_poses)
    print(f"✨ Exported poses GLB to {out_poses}")
    



if __name__ == "__main__":
    main()
