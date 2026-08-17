"""
run_train.py
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
  2. LLM-generated semantic styles          (--llm_api_key / --llm_styles 200)
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



# Dataset helpers (minimal)

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
    "domainnet": [
        "The_Eiffel_Tower", "The_Great_Wall_of_China", "The_Mona_Lisa",
        "aircraft_carrier", "airplane", "alarm_clock", "ambulance", "angel",
        "animal_migration", "ant", "anvil", "apple", "arm", "asparagus",
        "axe", "backpack", "banana", "bandage", "barn", "baseball",
        "baseball_bat", "basket", "basketball", "bat", "bathtub", "beach",
        "bear", "beard", "bed", "bee", "belt", "bench", "bicycle",
        "binoculars", "bird", "birthday_cake", "blackberry", "blueberry",
        "book", "boomerang", "bottlecap", "bowtie", "bracelet", "brain",
        "bread", "bridge", "broccoli", "broom", "bucket", "bulldozer",
        "bus", "bush", "butterfly", "cactus", "cake", "calculator",
        "calendar", "camel", "camera", "camouflage", "campfire", "candle",
        "cannon", "canoe", "car", "carrot", "castle", "cat", "ceiling_fan",
        "cell_phone", "cello", "chair", "chandelier", "church", "circle",
        "clarinet", "clock", "cloud", "coffee_cup", "compass", "computer",
        "cookie", "cooler", "couch", "cow", "crab", "crayon", "crocodile",
        "crown", "cruise_ship", "cup", "diamond", "dishwasher",
        "diving_board", "dog", "dolphin", "donut", "door", "dragon",
        "dresser", "drill", "drums", "duck", "dumbbell", "ear", "elbow",
        "elephant", "envelope", "eraser", "eye", "eyeglasses", "face",
        "fan", "feather", "fence", "finger", "fire_hydrant", "fireplace",
        "firetruck", "fish", "flamingo", "flashlight", "flip_flops",
        "floor_lamp", "flower", "flying_saucer", "foot", "fork", "frog",
        "frying_pan", "garden", "garden_hose", "giraffe", "goatee",
        "golf_club", "grapes", "grass", "guitar", "hamburger", "hammer",
        "hand", "harp", "hat", "headphones", "hedgehog", "helicopter",
        "helmet", "hexagon", "hockey_puck", "hockey_stick", "horse",
        "hospital", "hot_air_balloon", "hot_dog", "hot_tub", "hourglass",
        "house", "house_plant", "hurricane", "ice_cream", "jacket", "jail",
        "kangaroo", "key", "keyboard", "knee", "knife", "ladder", "lantern",
        "laptop", "leaf", "leg", "light_bulb", "lighter", "lighthouse",
        "lightning", "line", "lion", "lipstick", "lobster", "lollipop",
        "mailbox", "map", "marker", "matches", "megaphone", "mermaid",
        "microphone", "microwave", "monkey", "moon", "mosquito", "motorbike",
        "mountain", "mouse", "moustache", "mouth", "mug", "mushroom", "nail",
        "necklace", "nose", "ocean", "octagon", "octopus", "onion", "oven",
        "owl", "paint_can", "paintbrush", "palm_tree", "panda", "pants",
        "paper_clip", "parachute", "parrot", "passport", "peanut", "pear",
        "peas", "pencil", "penguin", "piano", "pickup_truck", "picture_frame",
        "pig", "pillow", "pineapple", "pizza", "pliers", "police_car",
        "pond", "pool", "popsicle", "postcard", "potato", "power_outlet",
        "purse", "rabbit", "raccoon", "radio", "rain", "rainbow", "rake",
        "remote_control", "rhinoceros", "rifle", "river", "roller_coaster",
        "rollerskates", "sailboat", "sandwich", "saw", "saxophone",
        "school_bus", "scissors", "scorpion", "screwdriver", "sea_turtle",
        "see_saw", "shark", "sheep", "shoe", "shorts", "shovel", "sink",
        "skateboard", "skull", "skyscraper", "sleeping_bag", "smiley_face",
        "snail", "snake", "snorkel", "snowflake", "snowman", "soccer_ball",
        "sock", "speedboat", "spider", "spoon", "spreadsheet", "square",
        "squiggle", "squirrel", "stairs", "star", "steak", "stereo",
        "stethoscope", "stitches", "stop_sign", "stove", "strawberry",
        "streetlight", "string_bean", "submarine", "suitcase", "sun",
        "swan", "sweater", "swing_set", "sword", "syringe", "t-shirt",
        "table", "teapot", "teddy-bear", "telephone", "television",
        "tennis_racquet", "tent", "tiger", "toaster", "toe", "toilet",
        "tooth", "toothbrush", "toothpaste", "tornado", "tractor",
        "traffic_light", "train", "tree", "triangle", "trombone", "truck",
        "trumpet", "umbrella", "underwear", "van", "vase", "violin",
        "washing_machine", "watermelon", "waterslide", "whale", "wheel",
        "windmill", "wine_bottle", "wine_glass", "wristwatch", "yoga",
        "zebra", "zigzag",
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
                   choices=["pacs", "vlcs", "officehome", "domainnet"])
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
    p.add_argument("--clipart_boost", action="store_true",
                   help="Add clipart-specific templates to ensemble (helps OfficeHome Clipart domain)")

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

    #Load CLIP  (Improvement 4)
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

    #Class names
    class_names = (args.class_names
                   or DATASET_CLASSES.get(args.dataset)
                   or DATASET_CLASSES["pacs"])
    print(f"[run_train] Dataset: {args.dataset}  |  Classes: {len(class_names)}")

    #Templates
    if args.clipart_boost:
        templates = args.templates or [
            "a {cls} in a {style} style",
            "a {style} illustration of a {cls}",
            "a {cls} drawn in {style} style",
            "a flat {style} graphic of a {cls}",
            "a {style} clipart of a {cls}",
        ]
        print(f"[run_train] Clipart boost: using {len(templates)} clipart-biased templates")
    else:
        templates = args.templates or [
            "a {cls} in a {style} style",
            "a {style} style of a {cls}",
            "a photo of a {cls} with {style} like style",
        ]
    print(f"[run_train] Model ensemble: {len(templates)} templates")

    #LLM API key
    llm_key = None if args.no_llm else (
        args.llm_api_key or os.environ.get("ANTHROPIC_API_KEY")
    )
    if llm_key:
        print("[run_train] Improvement 2: LLM style generation ENABLED")
    else:
        print("[run_train] Improvement 2: Using built-in vocabulary (no API key)")

    #Build trainer
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

    #Train
    history = trainer.fit(verbose=True)

    #Print summary
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
