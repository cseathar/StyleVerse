"""


  - Train each style vector INDIVIDUALLY for L iterations
  - Orthogonality diversity loss + content consistency loss
  - Stage 2: classifier with ArcFace scale=5, margin=0.5

Key fix: clip_model.float() converts CLIP to float32 for training,
         eliminating all Float/Half dtype mismatches.

Usage:
    python run_train_correct.py \
        --dataset domainnet \
        --data_root "D:/LLM/SFDG/DomainNet/domain_net" \
        --backbone ViT-B/16 \
        --K 80 \
        --style_iters 100 \
        --cls_epochs 100
"""

import argparse
import sys
import os
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))



# Dataset class names

DATASET_CLASSES = {
    "pacs": ["dog", "elephant", "giraffe", "guitar",
             "horse", "house", "person"],
    "vlcs": ["bird", "car", "chair", "dog", "person"],
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
    "RN50":     512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}

TEMPLATES = [
    "a {cls} in a {style} style",
    "a {style} style of a {cls}",
    "a photo of a {cls} with {style} like style",
]



# ArcFace Loss  (scale=5, margin=0.5)

class ArcFaceLoss(nn.Module):
    def __init__(self, feature_dim, num_classes, scale=5.0, margin=0.5):
        super().__init__()
        self.scale  = scale
        self.cos_m  = math.cos(margin)
        self.sin_m  = math.sin(margin)
        self.th     = math.cos(math.pi - margin)
        self.mm     = math.sin(math.pi - margin) * margin
        self.weight = nn.Parameter(torch.empty(feature_dim, num_classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        W       = F.normalize(self.weight, dim=0)
        cos     = features @ W
        sin     = (1 - cos.pow(2)).clamp(0).sqrt()
        phi     = cos * self.cos_m - sin * self.sin_m
        phi     = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits  = (one_hot * phi + (1 - one_hot) * cos) * self.scale
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def get_logits(self, features):
        W = F.normalize(self.weight, dim=0)
        return (features @ W) * self.scale



# Style-content encoding with gradient (all float32)

def encode_with_style_f32(clip_model, token_emb, tokens, si_norm, dummy_id):
    """
    Encode tokens with si injected at dummy_id position.
    CLIP model must be in float32 mode (call clip_model.float() first).
    All tensors are float32 — no dtype mismatch possible.
    Gradient flows: loss -> feat -> transformer -> x[:,pos,:] -> si_norm -> si
    """
    # Find dummy token position in first sequence
    row = tokens[0].tolist()
    try:
        pos = row.index(dummy_id)
    except ValueError:
        pos = min(4, tokens.shape[1] - 2)

    orig_scale = token_emb.weight.data[dummy_id].norm().item()

    # Get token embeddings — float32 (CLIP is in float32 mode)
    x = token_emb(tokens)                               # (B, 77, D) float32

    # Inject si at pos — gradient preserved via clone + assignment
    si_scaled = (si_norm * orig_scale).squeeze(0)       # (D,) float32
    x = x.clone()
    x[:, pos, :] = si_scaled.unsqueeze(0).expand(x.shape[0], -1)

    # All of CLIP is now float32 — no dtype conflict
    x = x + clip_model.positional_embedding             # float32 + float32
    x = x.permute(1, 0, 2)                             # (77, B, D)
    x = clip_model.transformer(x)                       # float32 throughout
    x = x.permute(1, 0, 2)                             # (B, 77, D)
    x = clip_model.ln_final(x)

    eot  = tokens.argmax(dim=-1)
    feat = x[torch.arange(x.shape[0]), eot]
    feat = feat @ clip_model.text_projection

    return F.normalize(feat, dim=-1)                    # (B, D) float32



# Stage 1: Train each style vector INDIVIDUALLY 

def train_stage1_individual(clip_model, tokenizer, class_names,
                             device, feature_dim, K, L, lr=0.002):
    """
    Train K style vectors one at a time for L iterations each.
    CLIP must already be in float32 mode.
    """
    print(f"\n[Stage 1] Training {K} style vectors individually")
    print(f"[Stage 1] {L} iters/vector x {K} vectors = {K*L:,} total iters")
    print(f"[Stage 1] CLIP in float32 mode — no dtype issues\n")

    token_emb = clip_model.token_embedding
    dummy_id  = tokenizer(["distinctive"]).to(device)[0, 1].item()

    # Pre-encode plain class content features (no gradient needed)
    print(f"[Stage 1] Pre-encoding {len(class_names)} content features...")
    content_feats = []
    with torch.no_grad():
        for cls in class_names:
            cls_txt = cls.replace("_", " ").replace("-", " ").lower()
            tok     = tokenizer([cls_txt]).to(device)
            feat    = clip_model.encode_text(tok)
            content_feats.append(F.normalize(feat, dim=-1))
    content_feats = torch.cat(content_feats, dim=0)     # (M, D) float32

    # Pre-tokenize style-content prompts
    sc_texts  = [
        f"a {cls.replace('_',' ').replace('-',' ').lower()} "
        f"in a distinctive style"
        for cls in class_names
    ]
    sc_tokens = tokenizer(sc_texts).to(device)          # (M, 77)

    learned_styles = []
    t0 = time.time()

    for k in range(K):
        # New learnable style vector — float32
        si  = nn.Parameter(torch.randn(feature_dim, device=device) * 0.02)
        opt = SGD([si], lr=lr, momentum=0.9)

        for iteration in range(L):
            opt.zero_grad()

            si_norm = F.normalize(si.unsqueeze(0), dim=-1)   # (1, D) float32

            # Style diversity loss: orthogonality vs previous
            if learned_styles:
                prev    = torch.stack(learned_styles, dim=0) # (k, D)
                sims    = (si_norm @ prev.T).squeeze(0)       # (k,)
                L_style = sims.pow(2).mean()
            else:
                L_style = si.new_zeros(1).squeeze()

            # Content consistency loss
            # Use small batch — gradient flows through transformer to si
            B       = min(32, len(class_names))
            sc_feat = encode_with_style_f32(
                clip_model, token_emb,
                sc_tokens[:B], si_norm, dummy_id
            )                                                  # (B, D)

            cos_sim   = (sc_feat * content_feats[:B]).sum(dim=-1)
            L_content = (1.0 - cos_sim).mean()

            # Combined prompt loss (same as PromptStyler)
            L_total = L_style + L_content
            L_total.backward()
            opt.step()

        with torch.no_grad():
            si_final = F.normalize(si.detach(), dim=-1)
        learned_styles.append(si_final)

        elapsed = time.time() - t0
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  Style {k+1:3d}/{K}  "
                  f"ortho={L_style.item():.4f}  "
                  f"content={L_content.item():.4f}  "
                  f"[{elapsed:.1f}s]")

    style_vectors = torch.stack(learned_styles, dim=0)  # (K, D)
    print(f"\n[Stage 1] All {K} style vectors learned.")

    # Diversity report
    with torch.no_grad():
        gram      = style_vectors @ style_vectors.T
        off_diag  = gram - torch.eye(K, device=device)
        diversity = 1.0 - off_diag.abs().mean().item()
    print(f"[Stage 1] Diversity score: {diversity:.4f} "
          f"(1.0 = perfect orthogonality)\n")

    return style_vectors



# Pre-encode all (class x style) features for Stage 2

@torch.no_grad()
def encode_all_features(clip_model, tokenizer, class_names,
                         style_vectors, template, device, feature_dim):
    token_emb = clip_model.token_embedding
    dummy_id  = tokenizer(["distinctive"]).to(device)[0, 1].item()
    orig_emb  = token_emb.weight.data[dummy_id].clone()
    K         = style_vectors.shape[0]

    prompts    = [
        template.replace("{cls}", cls.replace("_", " ").lower()
                        ).replace("{style}", "distinctive")
        for cls in class_names
    ]
    all_tokens = tokenizer(prompts).to(device)

    feats_list = []
    for k in range(K):
        sv        = F.normalize(style_vectors[k].unsqueeze(0), dim=-1)
        sv_scaled = (sv * orig_emb.norm()).squeeze(0)
        token_emb.weight.data[dummy_id] = sv_scaled

        bf = []
        for s in range(0, len(class_names), 64):
            f = clip_model.encode_text(all_tokens[s:s+64])
            bf.append(F.normalize(f, dim=-1))
        feats_list.append(torch.cat(bf, dim=0))

    token_emb.weight.data[dummy_id] = orig_emb

    stacked   = torch.stack(feats_list, dim=1)         # (M, K, D)
    features  = stacked.reshape(-1, feature_dim)       # (M*K, D)
    class_ids = torch.cat([
        torch.full((K,), ci, dtype=torch.long, device=device)
        for ci in range(len(class_names))
    ])
    return features, class_ids



# Stage 2: Train classifier with improvements

def train_stage2_classifier(clip_model, tokenizer, class_names,
                              device, feature_dim, style_vectors, args):
    from models.style_remover import EnhancedStyleRemover
    from losses.losses        import TotalLoss

    num_classes   = len(class_names)
    num_templates = len(TEMPLATES)

    removers   = nn.ModuleList([
        EnhancedStyleRemover(feature_dim, num_classes,
                             lambda_inter=0.5, resample_n=4)
        for _ in range(num_templates)
    ]).to(device)

    criterions = nn.ModuleList([
        TotalLoss(feature_dim, num_classes,
                  scale=args.scale, margin=args.margin,
                  lambda_inter=0.5, w_dual=0.5)
        for _ in range(num_templates)
    ]).to(device)

    optimizers = [
        SGD(list(r.parameters()) + list(c.parameters()),
            lr=args.lr, momentum=0.9)
        for r, c in zip(removers, criterions)
    ]
    schedulers = [
        CosineAnnealingLR(opt, T_max=args.cls_epochs)
        for opt in optimizers
    ]

    # Pre-encode all features once
    print(f"[Stage 2] Pre-encoding for {num_templates} templates...")
    template_data = []
    for t_idx, template in enumerate(TEMPLATES):
        features, class_ids = encode_all_features(
            clip_model, tokenizer, class_names,
            style_vectors, template, device, feature_dim
        )
        style_feats = features[:style_vectors.shape[0]]
        template_data.append((features, class_ids, style_feats))
        print(f"  Template {t_idx+1}/{num_templates}: "
              f"{features.shape[0]} samples")

    print(f"\n[Stage 2] Training classifier for {args.cls_epochs} epochs...")
    t0 = time.time()

    for epoch in range(1, args.cls_epochs + 1):
        for t_idx, (features, class_ids, style_feats) in enumerate(
            template_data
        ):
            opt       = optimizers[t_idx]
            remover   = removers[t_idx]
            criterion = criterions[t_idx]
            opt.zero_grad()

            # Gaussian resampling (Improvement 1)
            extra_f, extra_ids = remover.update_gaussian_and_resample(
                features, class_ids
            )
            if extra_f.shape[0] > 0:
                aug_f   = torch.cat([features, extra_f],    dim=0)
                aug_ids = torch.cat([class_ids, extra_ids], dim=0)
            else:
                aug_f, aug_ids = features, class_ids

            removed = F.normalize(remover(aug_f.detach()), dim=-1)

            loss_dict = criterion(
                removed_features = removed,
                style_features   = style_feats,
                class_ids        = aug_ids,
                labels           = aug_ids,
            )
            loss_dict["total"].backward()
            opt.step()

        for sched in schedulers:
            sched.step()

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{args.cls_epochs}  "
                  f"loss={loss_dict['total'].item():.4f}  "
                  f"cls={loss_dict['cls'].item():.4f}  "
                  f"[{elapsed:.1f}s]")

    # Save checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    path = "checkpoints/dpstylerpp_final.pt"
    torch.save({
        "removers":      [r.state_dict() for r in removers],
        "criterions":    [c.state_dict() for c in criterions],
        "style_vectors": style_vectors.cpu(),
        "class_names":   class_names,
        "feature_dim":   feature_dim,
    }, path)
    print(f"\n[Stage 2] Checkpoint saved -> {path}")



# Argument parser

def get_args():
    p = argparse.ArgumentParser(description="DPStyler++ Correct Training")
    p.add_argument("--dataset",     type=str,   default="domainnet",
                   choices=list(DATASET_CLASSES.keys()))
    p.add_argument("--data_root",   type=str,   default="./data")
    p.add_argument("--backbone",    type=str,   default="ViT-B/16",
                   choices=list(CLIP_FEATURE_DIMS.keys()))
    p.add_argument("--K",           type=int,   default=80)
    p.add_argument("--style_iters", type=int,   default=100,
                   help="Iterations per style vector (Stage 1)")
    p.add_argument("--style_lr",    type=float, default=0.002)
    p.add_argument("--cls_epochs",  type=int,   default=100,
                   help="Classifier epochs (Stage 2)")
    p.add_argument("--lr",          type=float, default=0.008)
    p.add_argument("--scale",       type=float, default=5.0,
                   help="ArcFace scale (PromptStyler=5.0)")
    p.add_argument("--margin",      type=float, default=0.5,
                   help="ArcFace margin (PromptStyler=0.5)")
    return p.parse_args()



# Main

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    try:
        import clip
    except ImportError:
        print("pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    feature_dim = CLIP_FEATURE_DIMS[args.backbone]
    print(f"[train] Loading CLIP {args.backbone} (dim={feature_dim})")
    clip_model, _ = clip.load(args.backbone, device=device)

    # KEY FIX: Convert CLIP to float32
    # CLIP loads in float16 on GPU by default.
    # Converting to float32 eliminates ALL Float/Half dtype mismatches
    # when backpropping through the transformer during Stage 1.
    clip_model = clip_model.float()
    print(f"[train] CLIP converted to float32 (eliminates dtype errors)")

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    class_names = DATASET_CLASSES[args.dataset]
    print(f"[train] Dataset: {args.dataset} | Classes: {len(class_names)}")

    print("\n" + "="*60)
    print("  DPStyler++ Correct Two-Stage Training")
    print(f"  Stage 1: {args.K} vectors x {args.style_iters} iters each")
    print(f"           = {args.K * args.style_iters:,} total iterations")
    print(f"  Stage 2: {args.cls_epochs} epochs")
    print(f"  ArcFace : scale={args.scale}, margin={args.margin}")
    print("="*60)

    Path("checkpoints").mkdir(exist_ok=True)

    # Stage 1: Learn style vectors individually
    style_vectors = train_stage1_individual(
        clip_model  = clip_model,
        tokenizer   = clip.tokenize,
        class_names = class_names,
        device      = device,
        feature_dim = feature_dim,
        K           = args.K,
        L           = args.style_iters,
        lr          = args.style_lr,
    )
    torch.save(style_vectors, "checkpoints/style_vectors.pt")
    print(f"[train] Style vectors saved -> checkpoints/style_vectors.pt")

    # Stage 2: Train classifier
    train_stage2_classifier(
        clip_model    = clip_model,
        tokenizer     = clip.tokenize,
        class_names   = class_names,
        device        = device,
        feature_dim   = feature_dim,
        style_vectors = style_vectors,
        args          = args,
    )

    print("\nTraining complete!")
    print("Run: python eval_domainnet1.py")


if __name__ == "__main__":
    main()
