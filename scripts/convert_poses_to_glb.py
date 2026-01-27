import trimesh
import numpy as np
import sys

def convert_ply_to_glb(ply_path, output_path):
    print(f"Loading PLY from {ply_path}...")
    # Load PLY as a point cloud
    # Note: da3_streaming saves camera_poses.ply as a Point Cloud where each point is a camera center
    # and color represents the chunk index.
    # To visualize "poses" (frustums), we ideally need orientation data (rotation matrices).
    # da3_streaming also saves 'camera_poses.txt' which contains 4x4 matrices.
    # Using txt is better for full pose visualization.
    
    # Check if txt exists
    txt_path = ply_path.replace(".ply", ".txt")
    import os
    if not os.path.exists(txt_path):
        print(f"Warning: {txt_path} not found. Using PLY points only.")
        mesh = trimesh.load(ply_path)
        mesh.export(output_path)
        return

    print(f"Loading poses from {txt_path}...")
    poses = []
    with open(txt_path, 'r') as f:
        for line in f:
            mat = np.array(list(map(float, line.strip().split()))).reshape(4, 4)
            poses.append(mat)
    
    print(f"Found {len(poses)} poses.")
    
    # Create a scene with camera frustums
    scene = trimesh.Scene()
    
    # A simple pyramid mesh to represent camera
    # Tip at origin, looking down +Z (or -Z depends on convention)
    # DA3 uses OpenCV convention? usually +Z forward.
    # We will create a generic frustum marker.
    marker = trimesh.creation.cone(radius=0.1, height=0.2, sections=4)
    # Rotate cone so tip is at origin and base is pointing out
    # Default cone is along Z, from 0 to height. Tip is at 0, 0, height?
    # trimesh cone: "height: The height of the cone, along the z-axis."
    # We want tip at camera center.
    
    # Let's use axis markers for clarity.
    
    for i, pose in enumerate(poses):
        # Create a camera marker
        cam = trimesh.creation.axis(origin_size=0.05, axis_length=0.2)
        cam.apply_transform(pose)
        scene.add_geometry(cam)
        
        # Add a point for the center
        # sphere = trimesh.creation.icosphere(radius=0.05)
        # sphere.apply_transform(pose)
        # scene.add_geometry(sphere)
        
    print(f"Exporting scene to {output_path}...")
    scene.export(output_path)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_poses.py <input.ply/txt> <output.glb>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # If input is ply, assume txt is next to it
    convert_ply_to_glb(input_path, output_path)
