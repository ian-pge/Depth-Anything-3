### 📦 Installation

#### Using Pixi (Recommended)
This project supports [Pixi](https://pixi.sh) for environment management, which handles both Conda and PyPI dependencies automatically.

```bash
# Initialize environment and install dependencies
pixi install

# Run the CLI
pixi run da3 --help
```

> [!NOTE]
> For GPU support, the `pixi.toml` is configured to use `conda-forge`'s `pytorch` with CUDA. If `pixi` fails to detect your GPU, ensure you have the appropriate drivers installed.

### 🚀 Running the models

#### Pose Extraction (Automated Pipeline)
To extract camera poses from a video sequence or image directory, use the automated pipeline:

```bash
# Basic usage
pixi run pose-extract --image_dir /path/to/images --output_dir /path/to/results

# Configure chunk size (smaller is better for low VRAM)
pixi run pose-extract --image_dir /path/to/images --output_dir /path/to/results --chunk_size 10 --overlap 5
```

This command will:
1.  Automatically download required weights (`DA3-LARGE`).
2.  Run the streaming pipeline which aligns chunks and closes loops.
3.  Export a `.glb` file for easy visualization.

#### Basic Usage (Pose & Depth)
```bash
# Process a directory of images
pixi run da3 images /path/to/images --export-dir /path/to/results --model-dir depth-anything/DA3NESTED-GIANT-LARGE
```

#### Memory-Efficient Streaming Inference (Video/Large Scenes)
For processing long sequences or large scenes with limited VRAM (e.g., < 24GB for the Giant model), use the streaming pipeline.

```bash
# 1. Download required weights (SALAD & DA3-Giant)
pixi run da3-weights

# 2. Run streaming inference
pixi run da3-streaming --image_dir /path/to/images --output_dir /path/to/results --config configs/base_config.yaml
```

> [!TIP]
> The streaming method uses a sliding window approach to maintain a constant memory footprint, allowing `DA3-Giant` to run on consumer hardware.

For detailed model information, please refer to the [Model Cards](#-model-cards) section below.

## 📚 Useful Documentation

- 🖥️ [Command Line Interface](docs/CLI.md)
- 📑 [Python API](docs/API.md)
- 📊 [Benchmark Evaluation](docs/BENCHMARK.md)

## 🗂️ Model Cards

Generally, you should observe that DA3-LARGE achieves comparable results to VGGT.

The Nested series uses an Any-view model to estimate pose and depth, and a monocular metric depth estimator for scaling. 

⚠️ Models with the `-1.1` suffix are retrained after fixing a training bug; prefer these refreshed checkpoints. The original `DA3NESTED-GIANT-LARGE`, `DA3-GIANT`, and `DA3-LARGE` remain available but are deprecated. You could expect much better performance for street scenes with the `-1.1` models.

| 🗃️ Model Name                  | 📏 Params | 📊 Rel. Depth | 📷 Pose Est. | 🧭 Pose Cond. | 🎨 GS | 📐 Met. Depth | ☁️ Sky Seg | 📄 License     |
|-------------------------------|-----------|---------------|--------------|---------------|-------|---------------|-----------|----------------|
| **Nested** | | | | | | | | |
| [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1)  | 1.40B     | ✅             | ✅            | ✅             | ✅     | ✅             | ✅         | CC BY-NC 4.0   |
| [DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE)  | 1.40B     | ✅             | ✅            | ✅             | ✅     | ✅             | ✅         | CC BY-NC 4.0   |
| **Any-view Model** | | | | | | | | |
| [DA3-GIANT-1.1](https://huggingface.co/depth-anything/DA3-GIANT-1.1)                     | 1.15B     | ✅             | ✅            | ✅             | ✅     |               |           | CC BY-NC 4.0   |
| [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT)                     | 1.15B     | ✅             | ✅            | ✅             | ✅     |               |           | CC BY-NC 4.0   |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1)                     | 0.35B     | ✅             | ✅            | ✅             |       |               |           | CC BY-NC 4.0     |
| [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE)                     | 0.35B     | ✅             | ✅            | ✅             |       |               |           | CC BY-NC 4.0     |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE)                     | 0.12B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
| [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL)                     | 0.08B     | ✅             | ✅            | ✅             |       |               |           | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Metric Depth** | | | | | | | | |
| [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE)              | 0.35B     | ✅             |              |               |       | ✅             | ✅         | Apache 2.0     |
|                               |           |               |              |               |               |       |           |                |
| **Monocular Depth** | | | | | | | | |
| [DA3MONO-LARGE](https://huggingface.co/depth-anything/DA3MONO-LARGE)                | 0.35B     | ✅             |              |               |               |       | ✅         | Apache 2.0     |


## ❓ FAQ

- **Monocular Metric Depth**: To obtain metric depth in meters from `DA3METRIC-LARGE`, use `metric_depth = focal * net_output / 300.`, where `focal` is the focal length in pixels (typically the average of fx and fy from the camera intrinsic matrix K). Note that the output from `DA3NESTED-GIANT-LARGE` is already in meters.

- <a id="use-ray-pose"></a>**Ray Head (`use_ray_pose`)**:  Our API and CLI support `use_ray_pose` arg, which means that the model will derive camera pose from ray head, which is generally slightly slower, but more accurate. Note that the default is `False` for faster inference speed. 
  <details>
  <summary>AUC3 Results for DA3NESTED-GIANT-LARGE</summary>
  
  | Model | HiRoom | ETH3D | DTU | 7Scenes | ScanNet++ | 
  |-------|------|-------|-----|---------|-----------|
  | `ray_head` | 84.4 | 52.6 | 93.9 | 29.5 | 89.4 |
  | `cam_head` | 80.3 | 48.4 | 94.1 | 28.5 | 85.0 |

  </details>
