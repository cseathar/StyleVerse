"""

Key differences from our previous approach:
  - Visual adapter (transformer + MLP) instead of text-side ArcFace
  - AdamW optimizer instead of SGD
  - 80 averaged ImageNet templates for text features
  - Trains on IMAGE features not text features
  - Cross-entropy loss instead of ArcFace

Usage:
    python run_train_adapter.py \
        --dataset domainnet \
        --data_root "D:\LLM\SFDG\DomainNet\domain_net" \
        --backbone ViT-B/16 \
        --iterations 15000
"""

import argparse
import sys
import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))


# 80 ImageNet templates (standard CLIP ensemble)

IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]



# Lightweight Visual Adapter 

class VisualAdapter(nn.Module):
    """
    Lightweight adapter: one transformer attention layer + MLP projector.
    Trained on image features to make them more domain-invariant.
    """

    def __init__(self, feature_dim: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim   = feature_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)

        # MLP projector
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)  →  add sequence dim for attention
        x_seq = x.unsqueeze(1)                        # (B, 1, D)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.norm1(x_seq + attn_out)          # residual
        mlp_out = self.mlp(x_seq)
        x_seq = self.norm2(x_seq + mlp_out)           # residual
        return x_seq.squeeze(1)                        # (B, D)



# Adapter + Classifier model
class AdapterClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.adapter    = VisualAdapter(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        x = self.adapter(x)
        x = F.normalize(x, dim=-1)
        return self.classifier(x)                      # (B, num_classes)



# Dataset class names

DATASET_CLASSES = {
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
    "RN50":     512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}


# Build text classifier from averaged templates

@torch.no_grad()
def build_text_classifier(clip_model, tokenizer, class_names,
                           device, feature_dim):
    """
    Build zero-shot text classifier by averaging 80 ImageNet templates.
    Returns (num_classes, feature_dim) weight matrix.
    """
    print(f"[adapter] Building text classifier from "
          f"{len(IMAGENET_TEMPLATES)} templates × {len(class_names)} classes...")

    all_weights = []
    for cls_name in class_names:
        # Format class name for prompt (replace underscores with spaces)
        cls_display = cls_name.replace("_", " ").replace("-", " ").lower()

        # Encode all templates for this class
        texts  = [t.format(cls_display) for t in IMAGENET_TEMPLATES]
        tokens = tokenizer(texts).to(device)

        # Encode in batches
        feats_list = []
        batch = 32
        for i in range(0, len(texts), batch):
            f = clip_model.encode_text(tokens[i:i+batch]).float()
            feats_list.append(F.normalize(f, dim=-1))

        feats = torch.cat(feats_list, dim=0)  # (80, D)
        mean_feat = feats.mean(dim=0)          # (D,)
        mean_feat = F.normalize(mean_feat, dim=-1)
        all_weights.append(mean_feat)

    weights = torch.stack(all_weights, dim=0)  # (num_classes, D)
    print(f"[adapter] Text classifier built. Shape: {weights.shape}")
    return weights



# Training

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[adapter] Device: {device}")

    try:
        import clip
    except ImportError:
        print("pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    feature_dim = CLIP_FEATURE_DIMS[args.backbone]
    print(f"[adapter] Loading CLIP {args.backbone} (dim={feature_dim})")
    clip_model, _ = clip.load(args.backbone, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    class_names = DATASET_CLASSES[args.dataset]
    num_classes  = len(class_names)
    print(f"[adapter] Dataset: {args.dataset} | Classes: {num_classes}")

    # Build text classifier weights
    text_weights = build_text_classifier(
        clip_model, clip.tokenize, class_names, device, feature_dim
    )  # (num_classes, D)

    # Build visual adapter + linear classifier
    model = AdapterClassifier(feature_dim, num_classes).to(device)

    # Initialize classifier with text weights (zero-shot warm start)
    with torch.no_grad():
        model.classifier.weight.copy_(text_weights)
    print(f"[adapter] Classifier initialized with text weights (zero-shot warm start)")

    # AdamW optimizer (key difference from our SGD approach)
    optimizer = AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.iterations)

    # Pre-generate style-augmented text features for training signal
    # Use our style generator for diversity
    print(f"[adapter] Pre-generating training features...")

    from models.style_generator import LLMStyleGenerator, StyleVectorGenerator

    llm_gen = LLMStyleGenerator(api_key=None)
    style_gen = StyleVectorGenerator(
        clip_model     = clip_model,
        tokenizer      = clip.tokenize,
        device         = device,
        feature_dim    = feature_dim,
        K              = args.K,
        llm_generator  = llm_gen,
        strategy_probs = (0.25, 0.25, 0.50),
    )
    style_gen.initialise()

    print("\n" + "="*55)
    print("  DPStyler++ Visual Adapter Training")
    print(f"  Backbone : {args.backbone}")
    print(f"  Classes  : {num_classes}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Optimizer: AdamW (lr={args.lr})")
    print("="*55 + "\n")

    model.train()
    t0        = time.time()
    iteration = 0
    best_loss = float("inf")

    while iteration < args.iterations:
        # Refresh style vectors every epoch-equivalent
        style_vectors = style_gen.refresh()  # (K, D)

        # Build training features: for each class, create K style-augmented features
        token_emb  = clip_model.token_embedding
        clip_dtype = token_emb.weight.dtype
        dummy_id   = clip.tokenize(["distinctive"]).to(device)[0, 1].item()
        orig_emb   = token_emb.weight.data[dummy_id].clone()

        # Encode all class prompts for K styles
        prompts    = [f"a {cn.replace('_',' ').lower()}" for cn in class_names]
        all_tokens = clip.tokenize(prompts).to(device)

        all_feats  = []
        all_labels = []

        with torch.no_grad():
            for k in range(args.K):
                sv = F.normalize(
                    style_vectors[k].unsqueeze(0), dim=-1
                ).to(clip_dtype)
                token_emb.weight.data[dummy_id] = (sv * orig_emb.norm()).squeeze(0)

                # Encode all classes with this style
                feats = clip_model.encode_text(all_tokens).float()
                feats = F.normalize(feats, dim=-1)
                all_feats.append(feats)
                all_labels.append(torch.arange(num_classes, device=device))

            token_emb.weight.data[dummy_id] = orig_emb

        features = torch.cat(all_feats,  dim=0)   # (num_classes*K, D)
        labels   = torch.cat(all_labels, dim=0)   # (num_classes*K,)

        # Shuffle
        perm     = torch.randperm(features.shape[0], device=device)
        features = features[perm]
        labels   = labels[perm]

        # Mini-batch training
        batch_size = min(args.batch_size, features.shape[0])
        for start in range(0, features.shape[0], batch_size):
            if iteration >= args.iterations:
                break

            batch_feats  = features[start:start+batch_size]
            batch_labels = labels[start:start+batch_size]

            optimizer.zero_grad()
            logits = model(batch_feats)
            loss   = F.cross_entropy(logits, batch_labels)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            iteration += 1

            if iteration % 500 == 0 or iteration == 1:
                elapsed = time.time() - t0
                print(f"  Iter {iteration:5d}/{args.iterations}  "
                      f"loss={loss.item():.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.6f}  "
                      f"[{elapsed:.1f}s]")

            if loss.item() < best_loss:
                best_loss = loss.item()

    # Save checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = "checkpoints/dpstylerpp_adapter_final.pt"
    torch.save({
        "adapter_state": model.state_dict(),
        "text_weights":  text_weights.cpu(),
        "class_names":   class_names,
        "feature_dim":   feature_dim,
        "backbone":      args.backbone,
    }, ckpt_path)
    print(f"\n[adapter] Checkpoint saved -> {ckpt_path}")
    print(f"[adapter] Best loss: {best_loss:.4f}")
    print("\nRun: python eval_domainnet_adapter.py")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="DPStyler++ Adapter Training")
    p.add_argument("--dataset",      type=str,   default="domainnet")
    p.add_argument("--data_root",    type=str,   default="./data")
    p.add_argument("--backbone",     type=str,   default="ViT-B/16")
    p.add_argument("--iterations",   type=int,   default=15000)
    p.add_argument("--K",            type=int,   default=80)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size",   type=int,   default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
