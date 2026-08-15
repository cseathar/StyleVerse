"""
 DomainNet Adapter Evaluation Script

Run: python eval_domainnet_adapter.py
"""

import multiprocessing
multiprocessing.freeze_support()

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
import clip
from torchvision import datasets
from torch.utils.data import DataLoader

DATA_ROOT  = "D:\\LLM+SFDG\\DomainNet\\domain_net"
CHECKPOINT = "checkpoints/dpstylerpp_adapter_final.pt"
BATCH_SIZE = 64
DOMAINS    = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

DPSTYLER_BASELINE = {
    "clipart":   71.5,
    "infograph": 49.1,
    "painting":  66.8,
    "quickdraw": 16.2,
    "real":      83.7,
    "sketch":    64.4,
    "average":   58.5,
}


class VisualAdapter(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        s = x.unsqueeze(1)
        a, _ = self.attention(s, s, s)
        s = self.norm1(s + a)
        s = self.norm2(s + self.mlp(s))
        return s.squeeze(1)


class AdapterClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.adapter    = VisualAdapter(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        x = self.adapter(x)
        x = F.normalize(x, dim=-1)
        return self.classifier(x)


def safe_loader(path):
    try:
        img = Image.open(path); img.load()
        return img.convert("RGB")
    except:
        return Image.new("RGB", (224, 224), (128, 128, 128))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}\n")

    # Load checkpoint
    ckpt        = torch.load(CHECKPOINT, map_location=device)
    class_names = ckpt["class_names"]
    feature_dim = ckpt["feature_dim"]
    backbone    = ckpt["backbone"]
    num_classes = len(class_names)

    print(f"[eval] Loading CLIP {backbone}...")
    clip_model, preprocess = clip.load(backbone, device=device)
    clip_model.eval()

    model = AdapterClassifier(feature_dim, num_classes).to(device)
    model.load_state_dict(ckpt["adapter_state"])
    model.eval()
    print(f"[eval] Model loaded. Classes: {num_classes}\n")

    print("=" * 62)
    print(f"  Evaluating DPStyler++ Adapter on DomainNet ({num_classes} classes)")
    print("=" * 62)

    results = {}
    for domain in DOMAINS:
        path = os.path.join(DATA_ROOT, domain)
        if not os.path.exists(path):
            print(f"  {domain}: NOT FOUND"); continue

        dataset = datasets.ImageFolder(path, transform=preprocess,
                                       loader=safe_loader)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

        correct = total = 0
        n = len(loader)
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(loader):
                if i % 100 == 0:
                    print(f"\r  {domain:<12} batch {i}/{n}...",
                          end="", flush=True)
                imgs   = imgs.to(device)
                labels = labels.to(device)
                feats  = clip_model.encode_image(imgs).float()
                logits = model(feats)
                preds  = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.shape[0]

        acc      = correct / total * 100.0
        baseline = DPSTYLER_BASELINE.get(domain, 0)
        diff     = acc - baseline
        sign     = "+" if diff >= 0 else ""
        results[domain] = acc
        print(f"\r  {domain:<12} {acc:6.2f}%   "
              f"(DPStyler: {baseline:.1f}%,  {sign}{diff:.2f}%)")

    avg          = sum(results.values()) / len(results)
    baseline_avg = DPSTYLER_BASELINE["average"]
    diff_avg     = avg - baseline_avg
    sign         = "+" if diff_avg >= 0 else ""

    print("-" * 62)
    print(f"  {'Average':<12}{avg:6.2f}%   "
          f"(DPStyler: {baseline_avg:.1f}%,  {sign}{diff_avg:.2f}%)")
    print("=" * 62)

    if avg > baseline_avg:
        print(f"\n  DPStyler++ BEATS DPStyler by {diff_avg:+.2f}% on DomainNet!")
    else:
        print(f"\n  Gap: {abs(diff_avg):.2f}% below DPStyler baseline.")


if __name__ == "__main__":
    main()
