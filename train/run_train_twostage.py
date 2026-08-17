"""
Two-Stage Training for Large Datasets

Stage 1: Learn good style vectors using diversity + content loss
Stage 2: Train classifier with those fixed style vectors + our improvements


Usage:
    python run_train_twostage.py \
        --dataset domainnet \
        --data_root "D:/LLM/SFDG/DomainNet/domain_net" \
        --backbone ViT-B/16 \
        --stage1_epochs 10 \
        --stage2_epochs 500
"""

import argparse
import sys
import os
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
    "pacs":   ["dog", "elephant", "giraffe", "guitar",
                "horse", "house", "person"],
    "vlcs":   ["bird", "car", "chair", "dog", "person"],
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


# Stage 1: Style Vector Learning 

class StyleVectorLearner:
    """
    Learns K style word vectors by optimising:
      L_style   = maximize pairwise cosine distance (diversity)
      L_content = minimize distance between style-content and content features
    """

    def __init__(self, clip_model, tokenizer, class_names,
                 device, feature_dim, K, epochs):
        self.clip_model  = clip_model
        self.tokenizer   = tokenizer
        self.class_names = class_names
        self.device      = device
        self.K           = K
        self.epochs      = epochs
        self.feature_dim = feature_dim

        # Learnable style word vectors
        self.style_vectors = nn.Parameter(
            torch.randn(K, feature_dim, device=device) * 0.02
        )
        self.optimizer = torch.optim.Adam([self.style_vectors], lr=0.002)

    @torch.no_grad()
    def _encode_content(self, class_name):
        """Encode plain class name."""
        tokens = self.tokenizer([class_name]).to(self.device)
        feat   = self.clip_model.encode_text(tokens).float()
        return F.normalize(feat, dim=-1)

    def _encode_style_content(self, class_name, style_vec):
        """Encode style+content prompt with style vector injection."""
        token_emb  = self.clip_model.token_embedding
        clip_dtype = token_emb.weight.dtype

        prompt    = f"a {class_name} in a distinctive style"
        tokens    = self.tokenizer([prompt]).to(self.device)

        dummy_tokens = self.tokenizer(["distinctive"]).to(self.device)
        dummy_id     = dummy_tokens[0, 1].item()

        original = token_emb.weight.data[dummy_id].clone()
        style_n  = F.normalize(style_vec.unsqueeze(0), dim=-1).to(clip_dtype)
        scale    = original.norm()
        token_emb.weight.data[dummy_id] = (style_n * scale).squeeze(0)

        feat = self.clip_model.encode_text(tokens).float()
        feat = F.normalize(feat, dim=-1)

        token_emb.weight.data[dummy_id] = original
        return feat

    def train(self):
        """Run Stage 1 style learning."""
        print(f"\n[Stage 1] Learning {self.K} style vectors "
              f"for {self.epochs} epochs...")

        # Pre-encode content features
        content_feats = []
        with torch.no_grad():
            for cls in self.class_names:
                content_feats.append(self._encode_content(cls))
        content_feats = torch.cat(content_feats, dim=0)  # (M, D)

        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()

            sv = F.normalize(self.style_vectors, dim=-1)  # (K, D)

            # Style diversity loss: maximise pairwise distances
            sim_matrix = sv @ sv.T                         # (K, K)
            mask       = ~torch.eye(self.K, dtype=torch.bool,
                                    device=self.device)
            L_style    = sim_matrix[mask].mean()           # minimise similarity

            # Content consistency loss: style+content close to content
            # Sample a subset of classes for speed
            n_sample  = min(20, len(self.class_names))
            cls_idxs  = torch.randperm(len(self.class_names))[:n_sample]
            k_idxs    = torch.randperm(self.K)[:10]

            content_loss_list = []
            for ci in cls_idxs:
                cls_name    = self.class_names[ci.item()]
                c_feat      = content_feats[ci]             # (D,)
                for ki in k_idxs:
                    sc_feat = self._encode_style_content(
                        cls_name, self.style_vectors[ki]
                    ).squeeze(0)                            # (D,)
                    # Style-content should be close to content
                    dist = 1.0 - (sc_feat * c_feat).sum()
                    content_loss_list.append(dist)

            L_content = torch.stack(content_loss_list).mean()
            L_total   = L_style + L_content

            L_total.backward()
            self.optimizer.step()

            if epoch % 2 == 0 or epoch == 1:
                print(f"  Stage1 Epoch {epoch:3d}/{self.epochs}  "
                      f"style={L_style.item():.4f}  "
                      f"content={L_content.item():.4f}")

        learned = F.normalize(self.style_vectors.detach(), dim=-1)
        print(f"[Stage 1] Done. Style vectors learned.\n")
        return learned


# Stage 2: Classifier Training with Fixed Style Vectors + Our Improvements

def train_stage2(clip_model, tokenizer, class_names, device,
                 feature_dim, style_vectors, args):
    
    from models.style_remover import EnhancedStyleRemover
    from losses.losses        import TotalLoss

    num_classes   = len(class_names)
    num_templates = 3
    templates     = [
        "a {cls} in a {style} style",
        "a {style} style of a {cls}",
        "a photo of a {cls} with {style} like style",
    ]

    removers   = nn.ModuleList([
        EnhancedStyleRemover(feature_dim, num_classes,
                             lambda_inter=args.lambda_inter)
        for _ in range(num_templates)
    ]).to(device)

    criterions = nn.ModuleList([
        TotalLoss(feature_dim, num_classes,
                  scale=args.scale, margin=args.margin,
                  lambda_inter=args.lambda_inter,
                  w_dual=args.w_dual)
        for _ in range(num_templates)
    ]).to(device)

    optimizers = [
        SGD(list(r.parameters()) + list(c.parameters()),
            lr=args.lr, momentum=0.9)
        for r, c in zip(removers, criterions)
    ]
    schedulers = [
        CosineAnnealingLR(opt, T_max=args.stage2_epochs)
        for opt in optimizers
    ]

    token_emb  = clip_model.token_embedding
    clip_dtype = token_emb.weight.dtype
    dummy_id   = clip_model.tokenizer(["distinctive"]).to(device)[0, 1].item() \
                 if hasattr(clip_model, 'tokenizer') else None

    # Pre-tokenize all class prompts for each template
    print(f"[Stage 2] Pre-encoding {num_templates * len(class_names)} "
          f"× {style_vectors.shape[0]} prompts...")

    # Pre-encode all features for all templates and styles
    all_template_features = []
    for t_idx, template in enumerate(templates):
        prompts = [
            template.replace("{cls}", cls).replace("{style}", "distinctive")
            for cls in class_names
        ]
        all_tokens = tokenizer(prompts).to(device)

        orig_emb   = token_emb.weight.data[
            tokenizer(["distinctive"]).to(device)[0, 1].item()
        ].clone()
        dummy_id_t = tokenizer(["distinctive"]).to(device)[0, 1].item()

        K          = style_vectors.shape[0]
        feats_list = []

        with torch.no_grad():
            for k in range(K):
                sv_norm = F.normalize(
                    style_vectors[k].unsqueeze(0), dim=-1
                ).to(clip_dtype)
                scale   = orig_emb.norm()
                token_emb.weight.data[dummy_id_t] = (sv_norm * scale).squeeze(0)
                feats   = clip_model.encode_text(all_tokens).float()
                feats   = F.normalize(feats, dim=-1)
                feats_list.append(feats)

            token_emb.weight.data[dummy_id_t] = orig_emb

        stacked  = torch.stack(feats_list, dim=1)    # (M, K, D)
        features = stacked.reshape(-1, feature_dim)  # (M*K, D)
        class_ids = torch.cat([
            torch.full((K,), ci, dtype=torch.long, device=device)
            for ci in range(len(class_names))
        ])
        all_template_features.append((features, class_ids))

    print(f"[Stage 2] Training classifier for {args.stage2_epochs} epochs...")

    import time
    t0 = time.time()

    for epoch in range(1, args.stage2_epochs + 1):
        for t_idx, (features, class_ids) in enumerate(all_template_features):
            opt = optimizers[t_idx]
            opt.zero_grad()

            # Gaussian update + resample (Improvement 1)
            remover = removers[t_idx]
            extra_feats, extra_ids = remover.update_gaussian_and_resample(
                features, class_ids
            )
            if extra_feats.shape[0] > 0:
                aug_features  = torch.cat([features,  extra_feats], dim=0)
                aug_class_ids = torch.cat([class_ids, extra_ids],   dim=0)
            else:
                aug_features, aug_class_ids = features, class_ids

            # Style removal
            removed = remover(aug_features.detach())
            removed = F.normalize(removed, dim=-1)

            # Style reference features (use first K style features)
            style_feats = all_template_features[0][0][:style_vectors.shape[0]]

            # Total loss
            criterion = criterions[t_idx]
            loss_dict = criterion(
                removed_features = removed,
                style_features   = style_feats,
                class_ids        = aug_class_ids,
                labels           = aug_class_ids,
            )

            loss_dict["total"].backward()
            opt.step()

        for sched in schedulers:
            sched.step()

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t0
            loss    = loss_dict["total"].item()
            cls_l   = loss_dict["cls"].item()
            print(f"  Stage2 Epoch {epoch:4d}/{args.stage2_epochs}  "
                  f"loss={loss:.4f}  cls={cls_l:.4f}  "
                  f"[{elapsed:.1f}s elapsed]")

    # Save checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = f"checkpoints/dpstylerpp_final.pt"
    torch.save({
        "removers":        [r.state_dict() for r in removers],
        "criterions":      [c.state_dict() for c in criterions],
        "style_vectors":   style_vectors.cpu(),
        "class_names":     class_names,
        "feature_dim":     feature_dim,
    }, ckpt_path)
    print(f"\n[Stage 2] Checkpoint saved -> {ckpt_path}")

    return removers, criterions


# Argument parser

def get_args():
    p = argparse.ArgumentParser(description="DPStyler++ Two-Stage Training")
    p.add_argument("--dataset",       type=str,   default="domainnet",
                   choices=list(DATASET_CLASSES.keys()))
    p.add_argument("--data_root",     type=str,   default="./data")
    p.add_argument("--backbone",      type=str,   default="ViT-B/16",
                   choices=list(CLIP_FEATURE_DIMS.keys()))
    p.add_argument("--stage1_epochs", type=int,   default=10)
    p.add_argument("--stage2_epochs", type=int,   default=500)
    p.add_argument("--K",             type=int,   default=80)
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--scale",         type=float, default=30.0)
    p.add_argument("--margin",        type=float, default=0.2)
    p.add_argument("--lambda_inter",  type=float, default=2.0)
    p.add_argument("--w_dual",        type=float, default=1.0)
    return p.parse_args()



# Main

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_train_twostage] Device: {device}")

    try:
        import clip
    except ImportError:
        print("ERROR: pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    feature_dim = CLIP_FEATURE_DIMS[args.backbone]
    print(f"[run_train_twostage] Loading CLIP {args.backbone} "
          f"(feature_dim={feature_dim})")
    clip_model, _ = clip.load(args.backbone, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    class_names = DATASET_CLASSES[args.dataset]
    print(f"[run_train_twostage] Dataset: {args.dataset} | "
          f"Classes: {len(class_names)}")

    print("\n" + "="*55)
    print("  DPStyler++ Two-Stage Training")
    print(f"  Stage 1: {args.stage1_epochs} epochs — learn style vectors")
    print(f"  Stage 2: {args.stage2_epochs} epochs — train classifier")
    print("="*55)

    # Stage 1: Learn style vectors
    learner = StyleVectorLearner(
        clip_model   = clip_model,
        tokenizer    = clip.tokenize,
        class_names  = class_names,
        device       = device,
        feature_dim  = feature_dim,
        K            = args.K,
        epochs       = args.stage1_epochs,
    )
    style_vectors = learner.train()   # (K, D) fixed learned vectors

    # Stage 2: Train classifier with improvements
    train_stage2(
        clip_model    = clip_model,
        tokenizer     = clip.tokenize,
        class_names   = class_names,
        device        = device,
        feature_dim   = feature_dim,
        style_vectors = style_vectors,
        args          = args,
    )

    print("\nTraining complete!")
    print("Run: python eval_domainnet.py")


if __name__ == "__main__":
    main()
