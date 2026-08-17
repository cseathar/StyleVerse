"""
 Innovation 1 Style Feature Resampling (SFR)
      Fit per-class Gaussian over M style features.
      Resample new features each epoch → infinite diversity.
      Formula: f̃ ~ N(μ̂ⱼ, Σ̂ⱼ) where
        μ̂ⱼ = (1/M) Σᵢ T(Pᵢⱼ)
        Σ̂ⱼ = (1/M-1) Σᵢ (T(Pᵢⱼ) - μ̂ⱼ)²

  Innovation 2 — Text Adapter
      F ∈ ℝ^{NK×D}: K=11 domain templates × N classes.
      Init: CLIP text encoder on "a [DOM] of a [CLS]".
      Kernel: φ(x) = exp(-β(1-x)), β=2.
      Logits = fW^T + α · φ(fF^T) · L
      Adapter LR = 0.001 for DomainNet.

  Style remover (SE-Net) before both.



Usage:
    # Full training:
    python run_train_correct.py \
        --dataset domainnet \
        --data_root "D:/LLM/SFDG/DomainNet/domain_net" \
        --backbone ViT-B/16 --K 80 \
        --style_iters 400 --cls_epochs 100

    # Skip Stage 1 (use saved style_vectors.pt):
    python run_train_correct.py \
        --dataset domainnet \
        --data_root "D:/LLM/SFDG/DomainNet/domain_net" \
        --backbone ViT-B/16 --cls_epochs 100 --skip_stage1
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

# Constants

# 11 domain templates for Text Adapter 
ADAPTER_DOMAINS_DOMAINNET = [
    "photo", "cartoon", "painting", "sketch", "clipart",
    "art", "drawing", "infograph", "quickdraw", "realistic", "vintage"
]

ADAPTER_DOMAINS_OFFICEHOME = [
    "art", "clipart", "product photo", "real world photo",
    "artistic", "digital art", "commercial product", "photograph",
    "illustration", "graphic design", "studio photo"
]

ADAPTER_DOMAINS_PACS = [
    "art painting", "cartoon", "photo", "sketch",
    "artistic", "illustrated", "photographic", "hand drawn",
    "painted", "animated", "realistic"
]

ADAPTER_DOMAINS_VLCS = [
    "photo", "realistic", "natural image", "color photograph",
    "digital photo", "outdoor photo", "indoor photo",
    "close up photo", "wide angle photo", "documentary", "snapshot"
]

DATASET_CLASSES = {
    "pacs": ["dog", "elephant", "giraffe", "guitar",
             "horse", "house", "person"],
    "vlcs": ["bird", "car", "chair", "dog", "person"],
    "officehome": [
        "Alarm_Clock","Backpack","Batteries","Bed","Bike","Bottle","Bucket",
        "Calculator","Calendar","Candles","Chair","Clipboards","Computer",
        "Couch","Curtains","Desk_Lamp","Drill","Eraser","Exit_Sign","Fan",
        "File_Cabinet","Flipflops","Flowers","Folder","Fork","Glasses",
        "Hammer","Helmet","Kettle","Keyboard","Knives","Lamp_Shade","Laptop",
        "Marker","Monitor","Mop","Mouse","Mug","Notebook","Oven","Pan",
        "Paper_Clip","Pen","Pencil","Postit_Notes","Printer","Push_Pin",
        "Radio","Refrigerator","Ruler","Scissors","Screwdriver","Shelf",
        "Sink","Sneakers","Soda","Speaker","Spoon","TV","Table","Telephone",
        "ToothBrush","Toys","Trash_Can","Webcam",
    ],
    "domainnet": [
        "The_Eiffel_Tower","The_Great_Wall_of_China","The_Mona_Lisa",
        "aircraft_carrier","airplane","alarm_clock","ambulance","angel",
        "animal_migration","ant","anvil","apple","arm","asparagus","axe",
        "backpack","banana","bandage","barn","baseball","baseball_bat",
        "basket","basketball","bat","bathtub","beach","bear","beard","bed",
        "bee","belt","bench","bicycle","binoculars","bird","birthday_cake",
        "blackberry","blueberry","book","boomerang","bottlecap","bowtie",
        "bracelet","brain","bread","bridge","broccoli","broom","bucket",
        "bulldozer","bus","bush","butterfly","cactus","cake","calculator",
        "calendar","camel","camera","camouflage","campfire","candle","cannon",
        "canoe","car","carrot","castle","cat","ceiling_fan","cell_phone",
        "cello","chair","chandelier","church","circle","clarinet","clock",
        "cloud","coffee_cup","compass","computer","cookie","cooler","couch",
        "cow","crab","crayon","crocodile","crown","cruise_ship","cup",
        "diamond","dishwasher","diving_board","dog","dolphin","donut","door",
        "dragon","dresser","drill","drums","duck","dumbbell","ear","elbow",
        "elephant","envelope","eraser","eye","eyeglasses","face","fan",
        "feather","fence","finger","fire_hydrant","fireplace","firetruck",
        "fish","flamingo","flashlight","flip_flops","floor_lamp","flower",
        "flying_saucer","foot","fork","frog","frying_pan","garden",
        "garden_hose","giraffe","goatee","golf_club","grapes","grass",
        "guitar","hamburger","hammer","hand","harp","hat","headphones",
        "hedgehog","helicopter","helmet","hexagon","hockey_puck",
        "hockey_stick","horse","hospital","hot_air_balloon","hot_dog",
        "hot_tub","hourglass","house","house_plant","hurricane","ice_cream",
        "jacket","jail","kangaroo","key","keyboard","knee","knife","ladder",
        "lantern","laptop","leaf","leg","light_bulb","lighter","lighthouse",
        "lightning","line","lion","lipstick","lobster","lollipop","mailbox",
        "map","marker","matches","megaphone","mermaid","microphone",
        "microwave","monkey","moon","mosquito","motorbike","mountain","mouse",
        "moustache","mouth","mug","mushroom","nail","necklace","nose","ocean",
        "octagon","octopus","onion","oven","owl","paint_can","paintbrush",
        "palm_tree","panda","pants","paper_clip","parachute","parrot",
        "passport","peanut","pear","peas","pencil","penguin","piano",
        "pickup_truck","picture_frame","pig","pillow","pineapple","pizza",
        "pliers","police_car","pond","pool","popsicle","postcard","potato",
        "power_outlet","purse","rabbit","raccoon","radio","rain","rainbow",
        "rake","remote_control","rhinoceros","rifle","river","roller_coaster",
        "rollerskates","sailboat","sandwich","saw","saxophone","school_bus",
        "scissors","scorpion","screwdriver","sea_turtle","see_saw","shark",
        "sheep","shoe","shorts","shovel","sink","skateboard","skull",
        "skyscraper","sleeping_bag","smiley_face","snail","snake","snorkel",
        "snowflake","snowman","soccer_ball","sock","speedboat","spider",
        "spoon","spreadsheet","square","squiggle","squirrel","stairs","star",
        "steak","stereo","stethoscope","stitches","stop_sign","stove",
        "strawberry","streetlight","string_bean","submarine","suitcase","sun",
        "swan","sweater","swing_set","sword","syringe","t-shirt","table",
        "teapot","teddy-bear","telephone","television","tennis_racquet","tent",
        "tiger","toaster","toe","toilet","tooth","toothbrush","toothpaste",
        "tornado","tractor","traffic_light","train","tree","triangle",
        "trombone","truck","trumpet","umbrella","underwear","van","vase",
        "violin","washing_machine","watermelon","waterslide","whale","wheel",
        "windmill","wine_bottle","wine_glass","wristwatch","yoga","zebra",
        "zigzag",
    ],
}

CLIP_FEATURE_DIMS = {"RN50": 512, "ViT-B/16": 512, "ViT-L/14": 768}

TEMPLATES = [
    "a {cls} in a {style} style",
    "a {style} style of a {cls}",
    "a photo of a {cls} with {style} like style",
]



# Stage 1: Individual Style Vector Training

def encode_with_style_f32(clip_model, token_emb, tokens, si_norm, dummy_id):
    """
    Encode tokens with si injected at dummy_id position.
    CLIP must be float32. Gradient flows back to si.
    """
    row = tokens[0].tolist()
    try:
        pos = row.index(dummy_id)
    except ValueError:
        pos = min(4, tokens.shape[1] - 2)

    orig_scale = token_emb.weight.data[dummy_id].norm().item()
    x = token_emb(tokens)                               # (B, 77, D) float32
    x = x.clone()
    x[:, pos, :] = (si_norm * orig_scale).squeeze(0
                    ).unsqueeze(0).expand(x.shape[0], -1)
    x = x + clip_model.positional_embedding
    x = clip_model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = clip_model.ln_final(x)
    eot  = tokens.argmax(dim=-1)
    feat = x[torch.arange(x.shape[0]), eot] @ clip_model.text_projection
    return F.normalize(feat, dim=-1)


def train_stage1_individual(clip_model, tokenizer, class_names,
                             device, feature_dim, K, L, lr=0.002):
    """
    Train K style vectors one-at-a-time for L iterations each.
    CLIP must already be float32.
    """
    print(f"\n[Stage 1] Individual training: K={K} vectors × {L} iters = "
          f"{K*L:,} total")

    token_emb = clip_model.token_embedding
    dummy_id  = tokenizer(["distinctive"]).to(device)[0, 1].item()

    # Pre-encode class content features
    print(f"[Stage 1] Pre-encoding {len(class_names)} content features...")
    content_feats = []
    with torch.no_grad():
        for cls in class_names:
            txt  = cls.replace("_", " ").replace("-", " ").lower()
            tok  = tokenizer([txt]).to(device)
            feat = clip_model.encode_text(tok)
            content_feats.append(F.normalize(feat, dim=-1))
    content_feats = torch.cat(content_feats, dim=0)    # (N, D)

    sc_texts  = [
        f"a {cls.replace('_',' ').replace('-',' ').lower()} in a distinctive style"
        for cls in class_names
    ]
    sc_tokens = tokenizer(sc_texts).to(device)

    learned = []
    t0 = time.time()

    for k in range(K):
        si  = nn.Parameter(torch.randn(feature_dim, device=device) * 0.02)
        opt = SGD([si], lr=lr, momentum=0.9)

        for iteration in range(L):
            opt.zero_grad()
            si_norm = F.normalize(si.unsqueeze(0), dim=-1)

            # Diversity: orthogonality vs all previously learned vectors
            if learned:
                prev    = torch.stack(learned, dim=0)
                sims    = (si_norm @ prev.T).squeeze(0)
                L_style = sims.pow(2).mean()
            else:
                L_style = si.new_zeros(1).squeeze()

            # Content consistency: small batch for speed
            B       = min(32, len(class_names))
            sc_feat = encode_with_style_f32(
                clip_model, token_emb, sc_tokens[:B], si_norm, dummy_id)
            L_content = (1.0 - (sc_feat * content_feats[:B]).sum(dim=-1)).mean()

            (L_style + L_content).backward()
            opt.step()

        with torch.no_grad():
            si_final = F.normalize(si.detach(), dim=-1)
        learned.append(si_final)

        if (k + 1) % 10 == 0 or k == 0:
            print(f"  Style {k+1:3d}/{K}  "
                  f"ortho={L_style.item():.4f}  "
                  f"content={L_content.item():.4f}  "
                  f"[{time.time()-t0:.1f}s]")

    style_vectors = torch.stack(learned, dim=0)         # (K, D)
    with torch.no_grad():
        off = (style_vectors @ style_vectors.T) - torch.eye(K, device=device)
        div = 1.0 - off.abs().mean().item()
    print(f"\n[Stage 1] Done. Diversity={div:.4f}  "
          f"(target > 0.90)\n")
    return style_vectors

# Innovation 1: Style Feature Resampling (SFR)
def compute_class_gaussians(features, class_ids, num_classes, feature_dim):
    """
    Fit per-class Gaussian from M style features.
    features:  (M*N, D)
    class_ids: (M*N,)
    Returns:   means (N, D), stds (N, D)
    """
    means = torch.zeros(num_classes, feature_dim, device=features.device)
    stds  = torch.zeros(num_classes, feature_dim, device=features.device)

    for c in range(num_classes):
        mask = (class_ids == c)
        if mask.sum() < 2:
            if mask.sum() == 1:
                means[c] = features[mask].squeeze(0)
            continue
        cls_feats = features[mask]               # (M, D)
        means[c]  = cls_feats.mean(0)
        stds[c]   = cls_feats.std(0).clamp(min=1e-6)

    return means, stds


@torch.no_grad()
def resample_features(means, stds, class_ids, n_resample=1):
    """
    Resample n_resample new features per original feature from class Gaussian.
    Returns resampled_feats (N_orig * n_resample, D), resampled_ids (N_orig * n_resample,)
    """
    N    = class_ids.shape[0]
    D    = means.shape[1]
    dev  = means.device

    all_feats = []
    all_ids   = []

    for i in range(N):
        c    = class_ids[i].item()
        mu   = means[c]    # (D,)
        sig  = stds[c]     # (D,)
        new  = mu + sig * torch.randn(n_resample, D, device=dev)
        new  = F.normalize(new, dim=-1)
        all_feats.append(new)
        all_ids.extend([c] * n_resample)

    return (torch.cat(all_feats, dim=0),
            torch.tensor(all_ids, dtype=torch.long, device=dev))

# Innovation 2: Text Adapter

class TextAdapter(nn.Module):
    """
    PromptTA Text Adapter.
    F ∈ ℝ^{NK×D}: pre-encoded domain-class features (trainable).
    L ∈ ℝ^{NK×N}: one-hot class labels (fixed).
    φ(x) = exp(-β(1-x)) — exponential kernel on cosine similarity.
    LogitsAdapter = φ(f @ F^T) @ L
    """
    def __init__(self, F_init: torch.Tensor, L: torch.Tensor,
                 beta: float = 2.0):
        super().__init__()
        self.F    = nn.Parameter(F_init.clone())   # (NK, D) — TRAINABLE
        self.register_buffer("L", L)               # (NK, N) — fixed
        self.beta = beta

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f: (B, D) normalized image/text features
        Returns: (B, N) logits
        """
        f_norm  = F.normalize(f, dim=-1)
        F_norm  = F.normalize(self.F, dim=-1)      # (NK, D)
        cos_sim = f_norm @ F_norm.T                # (B, NK)
        phi     = torch.exp(-self.beta * (1.0 - cos_sim))  # (B, NK)
        return phi @ self.L                        # (B, N)


@torch.no_grad()
def build_text_adapter(clip_model, tokenizer, class_names,
                        domains, device, feature_dim):
    """
    Build text adapter features F from "a [DOM] of a [CLS]" prompts.
    F: (K_dom * N_cls, D)
    L: (K_dom * N_cls, N_cls) one-hot
    """
    K_dom    = len(domains)
    N_cls    = len(class_names)
    print(f"[Adapter] Building F: {K_dom} domains × {N_cls} classes "
          f"= {K_dom * N_cls} prototypes...")

    F_feats  = []
    L_labels = []

    for dom_idx, dom in enumerate(domains):
        prompts    = [f"a {dom} of a {cls.replace('_',' ').lower()}"
                      for cls in class_names]
        all_tokens = tokenizer(prompts).to(device)

        bf = []
        for s in range(0, N_cls, 64):
            f = clip_model.encode_text(all_tokens[s:s+64])
            bf.append(F.normalize(f, dim=-1))
        dom_feats = torch.cat(bf, dim=0)           # (N_cls, D)
        F_feats.append(dom_feats)

        # One-hot labels for this domain's features
        one_hot = torch.zeros(N_cls, N_cls, device=device)
        one_hot.scatter_(1, torch.arange(N_cls, device=device).unsqueeze(1), 1.0)
        L_labels.append(one_hot)

    F_init = torch.cat(F_feats,  dim=0)   # (K_dom * N_cls, D)
    L_init = torch.cat(L_labels, dim=0)   # (K_dom * N_cls, N_cls)

    print(f"[Adapter] F shape: {F_init.shape}, L shape: {L_init.shape}\n")
    return F_init, L_init



# Pre-encode style features for Stage 2

@torch.no_grad()
def encode_all_features(clip_model, tokenizer, class_names,
                         style_vectors, template, device, feature_dim):
    token_emb = clip_model.token_embedding
    dummy_id  = tokenizer(["distinctive"]).to(device)[0, 1].item()
    orig_emb  = token_emb.weight.data[dummy_id].clone()
    K         = style_vectors.shape[0]
    N         = len(class_names)

    prompts    = [
        template.replace("{cls}", c.replace("_"," ").lower()
                        ).replace("{style}", "distinctive")
        for c in class_names
    ]
    all_tokens = tokenizer(prompts).to(device)
    feats_list = []

    for k in range(K):
        sv = F.normalize(style_vectors[k].unsqueeze(0), dim=-1)
        token_emb.weight.data[dummy_id] = (sv * orig_emb.norm()).squeeze(0)
        bf = []
        for s in range(0, N, 64):
            f = clip_model.encode_text(all_tokens[s:s+64])
            bf.append(F.normalize(f, dim=-1))
        feats_list.append(torch.cat(bf, dim=0))

    token_emb.weight.data[dummy_id] = orig_emb
    stacked   = torch.stack(feats_list, dim=1)        # (N, K, D)
    features  = stacked.reshape(-1, feature_dim)      # (N*K, D)
    class_ids = torch.cat([
        torch.full((K,), ci, dtype=torch.long, device=device)
        for ci in range(N)
    ])
    return features, class_ids

# Stage 2: Train with SFR + Text Adapter + Style Remover
def train_stage2(clip_model, tokenizer, class_names,
                  device, feature_dim, style_vectors, args):
    from models.style_remover import EnhancedStyleRemover
    from losses.losses        import TotalLoss

    num_classes   = len(class_names)
    num_templates = len(TEMPLATES)
    K             = style_vectors.shape[0]

    domain_map = {
        "domainnet": ADAPTER_DOMAINS_DOMAINNET,
        "officehome": ADAPTER_DOMAINS_OFFICEHOME,
        "pacs": ADAPTER_DOMAINS_PACS,
        "vlcs": ADAPTER_DOMAINS_VLCS,
    }
    adapter_doms = domain_map.get(args.dataset, ADAPTER_DOMAINS_DOMAINNET)
    F_init, L_mat = build_text_adapter(
        clip_model, tokenizer, class_names,
        adapter_doms, device, feature_dim
    )
    adapter  = TextAdapter(F_init, L_mat, beta=args.beta).to(device)

    # Style removers + classifiers
    removers   = nn.ModuleList([
        EnhancedStyleRemover(feature_dim, num_classes,
                             lambda_inter=args.lambda_inter, resample_n=4)
        for _ in range(num_templates)
    ]).to(device)

    criterions = nn.ModuleList([
        TotalLoss(feature_dim, num_classes,
                  scale=args.scale, margin=args.margin,
                  lambda_inter=args.lambda_inter, w_dual=args.w_dual)
        for _ in range(num_templates)
    ]).to(device)

    # Separate optimizers: classifier LR vs adapter LR (paper: 0.001)
    cls_params     = (list(r.parameters()) + list(c.parameters())
                      for r, c in zip(removers, criterions))
    adapter_params = list(adapter.parameters())

    cls_optimizers = [
        SGD(list(r.parameters()) + list(c.parameters()),
            lr=args.lr, momentum=0.9)
        for r, c in zip(removers, criterions)
    ]
    adapter_optimizer = SGD(adapter_params,
                            lr=args.adapter_lr, momentum=0.9)

    all_optimizers = cls_optimizers + [adapter_optimizer]
    schedulers     = [
        CosineAnnealingLR(opt, T_max=args.cls_epochs)
        for opt in all_optimizers
    ]

    # Pre-encode all (class × style) features
    print(f"[Stage 2] Pre-encoding for {num_templates} templates...")
    template_data = []
    for t_idx, template in enumerate(TEMPLATES):
        features, class_ids = encode_all_features(
            clip_model, tokenizer, class_names,
            style_vectors, template, device, feature_dim
        )
        template_data.append((features, class_ids))
        print(f"  Template {t_idx+1}: {features.shape[0]} samples")

    # Pre-compute class Gaussians for SFR (from first template as reference)
    ref_feats, ref_ids = template_data[0]
    means, stds = compute_class_gaussians(ref_feats, ref_ids,
                                           num_classes, feature_dim)
    print(f"[Stage 2] SFR: Gaussians computed for {num_classes} classes")
    print(f"\n[Stage 2] Training {args.cls_epochs} epochs "
          f"(classifier LR={args.lr}, adapter LR={args.adapter_lr})...")

    t0 = time.time()

    for epoch in range(1, args.cls_epochs + 1):

        # Innovation 1: Resample features at start of each epoch (SFR)
        sfr_feats, sfr_ids = resample_features(
            means, stds, ref_ids, n_resample=1)

        for t_idx, (features, class_ids) in enumerate(template_data):

            # Combine original + SFR resampled features
            aug_f   = torch.cat([features,  sfr_feats], dim=0)
            aug_ids = torch.cat([class_ids, sfr_ids],   dim=0)

            opt = cls_optimizers[t_idx]
            opt.zero_grad()
            adapter_optimizer.zero_grad()

            remover   = removers[t_idx]
            criterion = criterions[t_idx]

            # Gaussian resampling from style remover (DPStyler++ Improvement 1)
            extra_f, extra_ids = remover.update_gaussian_and_resample(
                aug_f, aug_ids)
            if extra_f.shape[0] > 0:
                aug_f   = torch.cat([aug_f,   extra_f],    dim=0)
                aug_ids = torch.cat([aug_ids, extra_ids],  dim=0)

            # Style removal
            removed = F.normalize(remover(aug_f.detach()), dim=-1)

            # Classifier loss (ArcFace)
            cls_loss_dict = criterion(
                removed_features=removed, style_features=features[:K],
                class_ids=aug_ids, labels=aug_ids)
            L_cls = cls_loss_dict["total"]

            # Innovation 2: Text adapter loss
            # Logits = fW^T + α · φ(fF^T) · L
            adapter_logits = adapter(removed.detach())
            L_adapter      = F.cross_entropy(adapter_logits, aug_ids)

            # Total loss
            L_total = L_cls + args.alpha * L_adapter
            L_total.backward()

            opt.step()
            adapter_optimizer.step()

        for sched in schedulers:
            sched.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.cls_epochs}  "
                  f"cls={cls_loss_dict['cls'].item():.4f}  "
                  f"adapter={L_adapter.item():.4f}  "
                  f"[{time.time()-t0:.1f}s]")

    # Save checkpoint (includes adapter)
    Path("checkpoints").mkdir(exist_ok=True)
    path = "checkpoints/dpstylerpp_final.pt"
    torch.save({
        "removers":      [r.state_dict() for r in removers],
        "criterions":    [c.state_dict() for c in criterions],
        "adapter":       adapter.state_dict(),
        "adapter_L":     L_mat.cpu(),
        "style_vectors": style_vectors.cpu(),
        "class_names":   class_names,
        "feature_dim":   feature_dim,
        "adapter_beta":  args.beta,
        "adapter_alpha": args.alpha,
        "adapter_domains": adapter_doms,
    }, path)
    print(f"\n[Stage 2] Checkpoint saved -> {path}")



# Args

def get_args():
    p = argparse.ArgumentParser(description="DPStyler++ + PromptTA")
    p.add_argument("--dataset",      type=str,   default="domainnet",
                   choices=list(DATASET_CLASSES.keys()))
    p.add_argument("--data_root",    type=str,   default="./data")
    p.add_argument("--backbone",     type=str,   default="ViT-B/16",
                   choices=list(CLIP_FEATURE_DIMS.keys()))
    # Stage 1
    p.add_argument("--K",            type=int,   default=80)
    p.add_argument("--style_iters",  type=int,   default=400,
                   help="Iterations per style vector — PromptStyler uses 400")
    p.add_argument("--style_lr",     type=float, default=0.002)
    p.add_argument("--skip_stage1",  action="store_true",
                   help="Load checkpoints/style_vectors.pt and skip Stage 1")
    # Stage 2
    p.add_argument("--cls_epochs",   type=int,   default=100)
    p.add_argument("--lr",           type=float, default=0.008,
                   help="Linear classifier LR")
    p.add_argument("--adapter_lr",   type=float, default=0.001,
                   help="Text adapter LR — PromptTA paper: 0.001 for DomainNet")
    p.add_argument("--scale",        type=float, default=5.0,
                   help="ArcFace scale — PromptStyler: 5.0")
    p.add_argument("--margin",       type=float, default=0.5,
                   help="ArcFace margin — PromptStyler: 0.5")
    p.add_argument("--lambda_inter", type=float, default=0.5)
    p.add_argument("--w_dual",       type=float, default=0.5)
    p.add_argument("--alpha",        type=float, default=3.0,
                   help="Adapter residual weight — PromptTA: 1-5")
    p.add_argument("--beta",         type=float, default=2.0,
                   help="Adapter kernel sharpness — PromptTA paper: 2")
    return p.parse_args()



# Main
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    try:
        import clip
    except ImportError:
        sys.exit("pip install git+https://github.com/openai/CLIP.git")

    feature_dim   = CLIP_FEATURE_DIMS[args.backbone]
    clip_model, _ = clip.load(args.backbone, device=device)
    clip_model     = clip_model.float()   # float32 — no dtype errors
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    class_names = DATASET_CLASSES[args.dataset]
    print(f"[train] {args.backbone} | {args.dataset} | "
          f"{len(class_names)} classes | float32")

    print("\n" + "="*65)
    print("  DPStyler++ + PromptTA Innovations")
    print(f"  [SFR] Style Feature Resampling — f̃ ~ N(μ̂ⱼ, Σ̂ⱼ)")
    print(f"  [TA ] Text Adapter — φ(fF^T)L, β={args.beta}, α={args.alpha}")
    print(f"  [TA ] Adapter LR = {args.adapter_lr} (PromptTA paper value)")
    print(f"  Stage 1: {args.K} vectors × {args.style_iters} iters")
    print(f"  Stage 2: {args.cls_epochs} epochs | "
          f"ArcFace scale={args.scale} margin={args.margin}")
    print("="*65)

    Path("checkpoints").mkdir(exist_ok=True)

    # Stage 1: Learn style vectors
    if args.skip_stage1:
        sv_path = "checkpoints/style_vectors.pt"
        if not os.path.exists(sv_path):
            sys.exit(f"ERROR: {sv_path} not found. "
                     "Run without --skip_stage1 first.")
        style_vectors = torch.load(sv_path, map_location=device)
        print(f"[train] Loaded style vectors: {style_vectors.shape}")
    else:
        style_vectors = train_stage1_individual(
            clip_model, clip.tokenize, class_names,
            device, feature_dim, args.K, args.style_iters, args.style_lr)
        torch.save(style_vectors, "checkpoints/style_vectors.pt")
        print("[train] Style vectors saved -> checkpoints/style_vectors.pt")

    #Stage 2: Train with SFR + Text Adapter 
    train_stage2(
        clip_model, clip.tokenize, class_names,
        device, feature_dim, style_vectors, args)

    print("\nTraining complete! Run: python eval_domainnet1.py")


if __name__ == "__main__":
    main()
