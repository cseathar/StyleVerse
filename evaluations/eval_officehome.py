

import multiprocessing
multiprocessing.freeze_support()

import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
import clip
from torchvision import datasets
from torch.utils.data import DataLoader
from models.style_remover import EnhancedStyleRemover
from losses.losses import TotalLoss


# CONFIG

DATA_ROOT   = "D:\\LLM+SFDG\\OfficeHome\\office_home"
CHECKPOINT  = "checkpoints/_final.pt"
BACKBONE    = "ViT-B/16"
FEATURE_DIM = 512
BATCH_SIZE  = 64

DOMAINS = ["Art", "Clipart", "Product", "Real_World"]

CLASS_NAMES = [
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
]

TEMPLATES = [
    "a {cls} in a {style} style",
    "a {style} style of a {cls}",
    "a photo of a {cls} with {style} like style",
]



# =============================================================================


def load_models(device):
    print(f"[eval] Loading CLIP {BACKBONE}...")
    clip_model, preprocess = clip.load(BACKBONE, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    num_classes   = len(CLASS_NAMES)
    num_templates = len(TEMPLATES)

    removers = nn.ModuleList([
        EnhancedStyleRemover(FEATURE_DIM, num_classes)
        for _ in range(num_templates)
    ])
    criterions = nn.ModuleList([
        TotalLoss(FEATURE_DIM, num_classes)
        for _ in range(num_templates)
    ])

    print(f"[eval] Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    for r, sd in zip(removers,   ckpt["removers"]):
        r.load_state_dict(sd)
    for c, sd in zip(criterions, ckpt["criterions"]):
        c.load_state_dict(sd)

    removers   = removers.to(device).eval()
    criterions = criterions.to(device).eval()
    print("[eval] Models loaded.\n")
    return clip_model, preprocess, removers, criterions


def predict_batch(img_feats, removers, criterions):
    img_feats  = F.normalize(img_feats, dim=-1)
    all_logits = []
    for remover, criterion in zip(removers, criterions):
        removed = F.normalize(remover(img_feats), dim=-1)
        logits  = criterion.cls_loss.get_logits(removed)
        all_logits.append(logits)
    stacked = torch.stack(all_logits, dim=0)
    max_logits, _ = stacked.max(dim=0)
    return max_logits.argmax(dim=-1)


def evaluate_domain(domain, clip_model, preprocess,
                    removers, criterions, device):
    path = os.path.join(DATA_ROOT, domain)
    if not os.path.exists(path):
        print(f"  [!] Not found: {path}")
        return None

    dataset = datasets.ImageFolder(root=path, transform=preprocess)

    # Verify class alignment
    if dataset.classes != CLASS_NAMES:
        print(f"\n  [!] WARNING: Class mismatch in {domain}!")
        print(f"      Expected: {CLASS_NAMES[:3]}...")
        print(f"      Got:      {dataset.classes[:3]}...")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=False)

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            feats  = clip_model.encode_image(images).float()
            preds  = predict_batch(feats, removers, criterions)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]

    return correct / total * 100.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}\n")

    # Verify domain folders exist
    print("Checking domain folders...")
    for d in DOMAINS:
        path = os.path.join(DATA_ROOT, d)
        exists = os.path.exists(path)
        print(f"  {d}: {'OK' if exists else 'NOT FOUND - check folder name!'}")
    print()

    clip_model, preprocess, removers, criterions = load_models(device)

    print("=" * 60)
    print("  Evaluating DPStyler++ on OfficeHome (65 classes)")
    print("=" * 60)

    results = {}
    for domain in DOMAINS:
        print(f"  {domain:<15}", end=" ", flush=True)
        acc = evaluate_domain(domain, clip_model, preprocess,
                              removers, criterions, device)
        if acc is not None:
            results[domain] = acc
            baseline = DPSTYLER_BASELINE.get(domain, 0)
            diff     = acc - baseline
            sign     = "+" if diff >= 0 else ""
            print(f"{acc:6.2f}%   "
                  f"(DPStyler: {baseline:.1f}%,  {sign}{diff:.2f}%)")
        else:
            print("SKIPPED")

    if not results:
        print("\n[!] No results.")
        print(f"    Check domain folders exist under: {DATA_ROOT}")
        return

    avg          = sum(results.values()) / len(results)
    baseline_avg = DPSTYLER_BASELINE["average"]
    diff_avg     = avg - baseline_avg
    sign         = "+" if diff_avg >= 0 else ""

    print("-" * 60)
    print(f"  {'Average':<15}{avg:6.2f}%   "
          f"(DPStyler: {baseline_avg:.1f}%,  {sign}{diff_avg:.2f}%)")
    print("=" * 60)

    if avg > baseline_avg:
        print(f"\n  DPStyler++ BEATS DPStyler by {diff_avg:+.2f}% on OfficeHome!")
    else:
        print(f"\n  Gap: {abs(diff_avg):.2f}% below DPStyler baseline.")


if __name__ == "__main__":
    main()
