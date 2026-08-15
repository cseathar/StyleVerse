
import multiprocessing; multiprocessing.freeze_support()
from PIL import ImageFile, Image; ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
import clip
from torchvision import datasets
from torch.utils.data import DataLoader
from models.style_remover import EnhancedStyleRemover
from losses.losses import TotalLoss

DATA_ROOT  = "D:\\LLM+SFDG\\OfficeHome/office_home"
CHECKPOINT = "checkpoints/dpstylerpp_final.pt"
BACKBONE   = "ViT-B/16"
FEATURE_DIM= 512
BATCH_SIZE = 64
DOMAINS    = ["Art", "Clipart", "Product", "Real_World"]

DPSTYLER_BASELINE = {
    "clipart":71.5,"infograph":49.1,"painting":66.8,
    "quickdraw":16.2,"real":83.7,"sketch":64.4,"average":58.5,
}

TEMPLATES = [
    "a {cls} in a {style} style",
    "a {style} style of a {cls}",
    "a photo of a {cls} with {style} like style",
]


class TextAdapter(nn.Module):
    def __init__(self, F_init, L, beta=2.0):
        super().__init__()
        self.F    = nn.Parameter(F_init.clone())
        self.register_buffer("L", L)
        self.beta = beta
    def forward(self, f):
        f_n = F.normalize(f, dim=-1)
        F_n = F.normalize(self.F, dim=-1)
        cos = f_n @ F_n.T
        phi = torch.exp(-self.beta * (1.0 - cos))
        return phi @ self.L


def safe_loader(path):
    try:
        img = Image.open(path); img.load(); return img.convert("RGB")
    except: return Image.new("RGB",(224,224),(128,128,128))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}\n")

    ckpt        = torch.load(CHECKPOINT, map_location=device)
    class_names = ckpt["class_names"]
    feature_dim = ckpt["feature_dim"]
    num_classes = len(class_names)

    print(f"[eval] Loading CLIP {BACKBONE}...")
    clip_model, preprocess = clip.load(BACKBONE, device=device)
    clip_model.eval()

    num_templates = len(TEMPLATES)
    removers   = nn.ModuleList([
        EnhancedStyleRemover(feature_dim, num_classes)
        for _ in range(num_templates)])
    criterions = nn.ModuleList([
        TotalLoss(feature_dim, num_classes)
        for _ in range(num_templates)])

    for r, sd in zip(removers, ckpt["removers"]):   r.load_state_dict(sd)
    for c, sd in zip(criterions, ckpt["criterions"]): c.load_state_dict(sd)
    removers   = removers.to(device).eval()
    criterions = criterions.to(device).eval()

    # Load text adapter if present
    has_adapter = "adapter" in ckpt
    adapter     = None
    alpha       = 0.0
    if has_adapter:
        F_init = ckpt["adapter"]["F"]
        L_mat  = ckpt["adapter_L"].to(device)
        beta   = ckpt.get("adapter_beta", 2.0)
        alpha  = ckpt.get("adapter_alpha", 3.0)
        adapter = TextAdapter(F_init, L_mat, beta).to(device).eval()
        adapter.load_state_dict(ckpt["adapter"])
        print(f"[eval] Text adapter loaded (α={alpha}, β={beta})")
    print(f"[eval] Models loaded. Classes: {num_classes}\n")

    print("=" * 62)
    print(f"  Evaluating DPStyler++ on DomainNet ({num_classes} classes)")
    print("=" * 62)

    results = {}
    for domain in DOMAINS:
        path = os.path.join(DATA_ROOT, domain)
        if not os.path.exists(path):
            print(f"  {domain}: NOT FOUND"); continue

        ds     = datasets.ImageFolder(path, transform=preprocess,
                                      loader=safe_loader)
        loader = DataLoader(ds, batch_size=BATCH_SIZE,
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
                feats  = F.normalize(feats, dim=-1)

                # Ensemble over templates
                all_logits = []
                for remover, criterion in zip(removers, criterions):
                    removed = F.normalize(remover(feats), dim=-1)
                    logits  = criterion.cls_loss.get_logits(removed)
                    all_logits.append(logits)

                stacked = torch.stack(all_logits, dim=0)
                final_logits, _ = stacked.max(dim=0)          # (B, N)

                # Add text adapter (PromptTA innovation 2)
                if adapter is not None:
                    # Use first remover's output for adapter
                    removed_0 = F.normalize(removers[0](feats), dim=-1)
                    adp_logits = adapter(removed_0)
                    final_logits = final_logits + alpha * adp_logits

                preds = final_logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.shape[0]

        acc      = correct / total * 100.0
        baseline = DPSTYLER_BASELINE.get(domain, 0)
        diff     = acc - baseline
        sign     = "+" if diff >= 0 else ""
        results[domain] = acc
        print(f"\r  {domain:<12} {acc:6.2f}%   "
              f"(DPStyler: {baseline:.1f}%,  {sign}{diff:.2f}%)")

    avg      = sum(results.values()) / len(results)
    baseline = DPSTYLER_BASELINE["average"]
    diff     = avg - baseline
    sign     = "+" if diff >= 0 else ""
    print("-" * 62)
    print(f"  {'Average':<12}{avg:6.2f}%   "
          f"(DPStyler: {baseline:.1f}%,  {sign}{diff:.2f}%)")
    print("=" * 62)
    if avg > baseline:
        print(f"\n  DPStyler++ BEATS DPStyler by {diff:+.2f}% on DomainNet!")
    else:
        print(f"\n  Gap: {abs(diff):.2f}% below DPStyler baseline.")


if __name__ == "__main__":
    main()
