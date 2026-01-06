#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from typing import List

import torch

# If your package lives somewhere non-standard, keep this.
# Otherwise you can remove it.
sys.path.append("/workspace/src")

from depth_anything_3.api import DepthAnything3


def find_images(input_dir: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    paths.sort()
    return paths


def load_model(model_name: str, device: torch.device):
    """
    Prefer from_pretrained (more stable), but fall back to constructor
    depending on which DepthAnything3 version you have installed.
    """
    # If you pass "da3-giant", we map to the HF id many installs expect.
    # If your install already accepts "da3-giant", it will still work in fallback.
    hf_id_map = {
        "da3-giant": "depth-anything/da3-giant",
        "da3nested-giant-large": "depth-anything/da3nested-giant-large",
    }
    hf_id = hf_id_map.get(model_name, model_name)

    # Try from_pretrained first
    if hasattr(DepthAnything3, "from_pretrained"):
        try:
            print(f"[INFO ] Loading model via from_pretrained: {hf_id}")
            model = DepthAnything3.from_pretrained(hf_id).to(device)
            return model
        except Exception as e:
            print(
                f"[WARN ] from_pretrained failed ({e}). Falling back to constructor..."
            )

    print(f"[INFO ] Loading model via constructor: model_name={model_name}")
    model = DepthAnything3(model_name=model_name).to(device)
    return model


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = find_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No images found in: {args.input_dir}")

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO ] Found {len(image_paths)} images in {args.input_dir}")
    print(f"[INFO ] Output dir: {args.output_dir}")
    print(f"[INFO ] Device: {device}")

    model = load_model(args.model_name, device)

    # --- Inference ---
    # Keep this minimal first; extra knobs can reduce stability on some sequences.
    infer_kwargs = dict(
        export_dir=args.output_dir,
        export_format=args.export_format,  # default "gs_ply"
        infer_gs=True,  # GS head enabled
    )

    # Optional knobs
    if args.align_to_input_ext_scale:
        infer_kwargs["align_to_input_ext_scale"] = True
    if args.ref_view_strategy is not None:
        infer_kwargs["ref_view_strategy"] = args.ref_view_strategy

    print("[INFO ] Running inference...")
    try:
        # Some versions want positional list, some want keyword `image=...`.
        try:
            prediction = model.inference(image_paths, **infer_kwargs)
        except TypeError:
            prediction = model.inference(image=image_paths, **infer_kwargs)
    except Exception as e:
        print(f"[ERROR] inference failed: {e}")
        raise

    print("[INFO ] Inference returned. Dumping sanity stats...")

    print(f"✅ Done. Results saved to: {args.output_dir}")
    print("   Look for .ply / .npz / .glb depending on export_format.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="/root/DA3_gcloud/example_images")
    p.add_argument("--output-dir", default="/root/instant_splat_giant")
    p.add_argument(
        "--model-name",
        default="da3-giant",
        help="Use 'da3-giant' (recommended for GS) or 'da3nested-giant-large'",
    )
    p.add_argument(
        "--export-format",
        default="gs_ply",
        help="Common: gs_ply or gs_ply-gs_video (if supported by your install)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images (0 = no limit). Try 10-20 for debugging overlap.",
    )
    p.add_argument(
        "--align-to-input-ext-scale",
        action="store_true",
        help="Optional: can help, but disable if you get empty-depth issues.",
    )
    p.add_argument(
        "--ref-view-strategy",
        default=None,
        help="Optional: e.g. saddle_balanced. Leave unset for defaults while debugging.",
    )
    args = p.parse_args()

    run(args)
