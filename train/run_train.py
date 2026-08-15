"""
 run_train.py.

Quick start
-----------
  python run_train.py \
      --dataset pacs \
      --data_root /path/to/data \
      --backbone ViT-B/16 \
      --epochs 100 \
      --llm_api_key sk-ant-...   # optional: falls back to built-in vocab

The four improvements are all active by default:
  1. Gaussian distribution style modelling  (--resample_n 4)
  2. LLM-generated semantic styles          (--llm_api_key )
  3. Dual-objective channel refinement loss (--lambda_inter 0.5)
  4. ViT-B/16 backbone                      (--backbone ViT-B/16)
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F

# Make local imports work
sys.path.insert(0, str(Path(__file__).parent))


# Dataset helpers 

DATASET_CLASSES = {
    "pacs":       ["dog", "elephant", "giraffe", "guitar",
                   "horse", "house", "person"],
    "vlcs":       ["bird", "car", "chair", "dog", "person"],
    "officehome": [
        "Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike",
        "Bottle", "Bucket", "Calculator", "Calendar", "Candles",
        "Chair", "Clipboards", "Computer", "Couch", "Curtains",
        "Desk_Lamp", "Drill", "Eraser", "Exit_Sign", "Fan",
        "File_Cabinet", "Flipflops", "Flowers", "Folder", "Fork",
        "Glasses", "Hammer", "Helmet", "Kettle", "Keyboard",
        "Knives", "Lamp_Shade", "Laptop", "Marker", "Monitor",
        "Mop", "Mouse", "Mug", "Notebook", "Oven",
        "Pan", "Paper_Clip", "Pen", "Pencil", "Postit_Notes",
        "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler",
        "Scissors", "Screwdriver", "Shelf", "Sink", "Sneakers",
        "Soda", "Speaker", "Spoon", "TV", "Table",
        "Telephone", "ToothBrush", "Toys", "Trash_Can", "Webcam",
    ],
}

CLIP_FEATURE_DIMS = {
    "RN50":      512,
    "ViT-B/16":  512,   # CLIP ViT-B/16 outputs 512-dim features
    "ViT-L/14":  768,
}



# Argument parser

def get_args():
    p = argparse.ArgumentParser(description="DPStyler++ Training")

    # Data
    p.add_argument("--dataset",   type=str, default="pacs",
                   choices=list(DATASET_CLASSES.keys()))
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--class_names", nargs="+", default=None,
                   help="Override class names (useful for custom datasets)")

    # Model / backbone
    p.add_argument("--backbone",  type=str, default="ViT-B/16",
                   choices=list(CLIP_FEATURE_DIMS.keys()),
                   help="CLIP backbone. ViT-B/16 or ViT-L/14 for Improvement 4")

    # Training
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--lr",        type=float, default=0.008)
    p.add_argument("--momentum",  type=float, default=0.9)
    p.add_argument("--K",         type=int,   default=80,
                   help="Number of style vectors per epoch")
    p.add_argument("--batch_size",type=int,   default=128)

    # Improvement 1: Gaussian resampling
    p.add_argument("--resample_n",type=int,   default=4,
                   help="[Improvement 1] Gaussian resample count per class per epoch")
    p.add_argument("--gaussian_momentum", type=float, default=0.1,
                   help="[Improvement 1] Momentum for Gaussian running stats")

    # Improvement 2: LLM styles
    p.add_argument("--llm_api_key",   type=str, default=None,
                   help="[Improvement 2] Anthropic API key for LLM style generation")
    p.add_argument("--llm_styles",    type=int, default=200,
                   help="[Improvement 2] Number of LLM-generated styles to request")
    p.add_argument("--llm_cache",     type=str, default="llm_styles_cache.json",
                   help="[Improvement 2] Cache path for LLM styles")
    p.add_argument("--no_llm",        action="store_true",
                   help="[Improvement 2] Disable LLM, use built-in vocabulary only")

    # Improvement 3: Dual-objective loss
    p.add_argument("--lambda_inter",  type=float, default=0.5,
                   help="[Improvement 3] Inter-class variance weight in dual loss")
    p.add_argument("--w_domain",      type=float, default=1.0)
    p.add_argument("--w_dual",        type=float, default=0.5,
                   help="[Improvement 3] Weight of dual-objective loss")
    p.add_argument("--w_cls",         type=float, default=1.0)

    # ArcFace
    p.add_argument("--scale",    type=float, default=5.0)
    p.add_argument("--margin",   type=float, default=0.5)

    # Templates (model ensemble)
    p.add_argument("--templates", nargs="+", default=None,
                   help="Text templates (one model trained per template)")

    # Output
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()



# Main

def main():
    args = get_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_train] Device: {device}")

    #  Load CLIP 
    try:
        import clip
    except ImportError:
        print("ERROR: CLIP not installed.  Run:  pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    print(f"[run_train] Loading CLIP backbone: {args.backbone}")
    clip_model, _ = clip.load(args.backbone, device=device)
    clip_model.eval()
    feature_dim = CLIP_FEATURE_DIMS[args.backbone]
    print(f"[run_train] Feature dimension: {feature_dim}")

    #  Class names
    class_names = (args.class_names
                   or DATASET_CLASSES.get(args.dataset)
                   or DATASET_CLASSES["pacs"])
    print(f"[run_train] Dataset: {args.dataset}  |  Classes: {len(class_names)}")

    # Templates 
    templates = args.templates or [
        "a {cls} in a {style} style",
        "a {style} style of a {cls}",
        "a photo of a {cls} with {style} like style",
    ]
    print(f"[run_train] Model ensemble: {len(templates)} templates")

    #  LLM API key 
    llm_key = None if args.no_llm else (
        args.llm_api_key or os.environ.get("ANTHROPIC_API_KEY")
    )
    if llm_key:
        print("[run_train] Improvement 2: LLM style generation ENABLED")
    else:
        print("[run_train] Improvement 2: Using built-in vocabulary (no API key)")

    # Build trainer
    from models.trainer import Trainer

    trainer = Trainer(
        clip_model      = clip_model,
        tokenizer       = clip.tokenize,
        class_names     = class_names,
        device          = device,
        feature_dim     = feature_dim,
        epochs          = args.epochs,
        lr              = args.lr,
        momentum        = args.momentum,
        K               = args.K,
        resample_n      = args.resample_n,
        lambda_inter    = args.lambda_inter,
        scale           = args.scale,
        margin          = args.margin,
        templates       = templates,
        llm_api_key     = llm_key,
        checkpoint_dir  = args.checkpoint_dir,
    )

    print("\n" + "="*60)
    print("  DPStyler++ Training")
    print("  Improvements active:")
    print("  [1] Gaussian Distribution Style Modelling")
    print("  [2] LLM-Generated Semantic Styles")
    print("  [3] Dual-Objective Channel Refinement Loss")
    print(f"  [4] {args.backbone} backbone  (feature_dim={feature_dim})")
    print("="*60 + "\n")

    #  Train 
    history = trainer.fit(verbose=True)

    #  Print summary 
    final = history[-1]
    print(f"\nTraining complete.")
    print(f"Final epoch losses → "
          f"total={final['total']:.4f}  "
          f"domain={final['domain']:.4f}  "
          f"dual={final['dual']:.4f}  "
          f"cls={final['cls']:.4f}")
    print(f"\nCheckpoint saved in: {args.checkpoint_dir}/")
    print("\nTo run inference:")
    print("  trainer.evaluate(test_loader, clip_model)")


if __name__ == "__main__":
    main()
