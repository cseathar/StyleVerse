import multiprocessing
multiprocessing.freeze_support()

import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
import clip
from torchvision import datasets
from torch.utils.data import DataLoader
from models.style_remover import EnhancedStyleRemover
from losses.losses import TotalLoss

DATA_ROOT  = "D:\\LLM+SFDG\\PACS"
CHECKPOINT = "checkpoints/_final.pt"
DOMAINS    = ["art_painting", "cartoon", "photo", "sketch"]
CLASS_NAMES = ["dog","elephant","giraffe","guitar","horse","house","person"]
TEMPLATES  = [
    "a {cls} in a {style} style",
    "a {style} style of a {cls}",
    "a photo of a {cls} with {style} like style",
]
BASELINE = {"art_painting":93.2,"cartoon":98.3,"photo":99.8,"sketch":82.4}

def main():
    device = torch.device("cuda")
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    removers   = nn.ModuleList([EnhancedStyleRemover(512, 7) for _ in TEMPLATES])
    criterions = nn.ModuleList([TotalLoss(512, 7) for _ in TEMPLATES])

    ckpt = torch.load(CHECKPOINT, map_location=device)
    for r, sd in zip(removers,   ckpt["removers"]):   r.load_state_dict(sd)
    for c, sd in zip(criterions, ckpt["criterions"]): c.load_state_dict(sd)
    removers.to(device).eval()
    criterions.to(device).eval()

    results = {}
    for domain in DOMAINS:
        path    = os.path.join(DATA_ROOT, domain)
        dataset = datasets.ImageFolder(path, transform=preprocess)
        loader  = DataLoader(dataset, batch_size=64,
                             shuffle=False, num_workers=0)  # 0 = Windows fix
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = F.normalize(clip_model.encode_image(imgs).float(), dim=-1)
                logits_all = []
                for r, c in zip(removers, criterions):
                    out = F.normalize(r(feats), dim=-1)
                    logits_all.append(c.cls_loss.get_logits(out))
                preds = torch.stack(logits_all).max(0)[0].argmax(-1)
                correct += (preds == labels).sum().item()
                total   += labels.shape[0]
        acc = correct / total * 100
        results[domain] = acc
        print(f"{domain}: {acc:.2f}%  (baseline {BASELINE[domain]:.1f}%)")

    avg = sum(results.values()) / len(results)
    print(f"\nAverage: {avg:.2f}%  (baseline 93.4%)")
    print(f"vs DPStyler: {avg - 93.4:+.2f}%")

if __name__ == "__main__":
    main()