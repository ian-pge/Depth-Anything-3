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
```

**Available Flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--image_dir` | **(Required)** Path to input images directory. | |
| `--output_dir` | **(Required)** Path to output directory. | |
| `--chunk_size` | Number of images per segment. Smaller values reduce VRAM usage. | `20` |
| `--overlap` | Number of overlapping images between segments. | `10` |
| `--no_loop` | Disable global loop closure detection. | Loop Enabled |
| `--align_method` | Method for chunk alignment (`sim3`, `se3`, or `scale+se3`). | `sim3` |
| `--depth_threshold` | Maximum depth value used for alignment (meters). | `15.0` |

This command will:
1.  Automatically download required weights (`DA3-LARGE`).
2.  Run the streaming pipeline which aligns chunks and closes loops.
3.  Export a `.glb` file for easy visualization of both camera poses and the 3D scene.

## 📚 Useful Documentation

- 🖥️ [Command Line Interface](docs/CLI.md)
- 📑 [Python API](docs/API.md)
- 📊 [Benchmark Evaluation](docs/BENCHMARK.md)

