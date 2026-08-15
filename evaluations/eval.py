
import torch
import clip
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, ".")
from models.trainer import Trainer

device = torch.device("cuda")
DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
DATA_ROOT = r"D:\LLM+SFDG\PACS"
CLASS_NAMES = ["dog","elephant","giraffe","guitar","horse","house","person"]

clip_model, preprocess = clip.load("ViT-B/16", device=device)

results = {}
for test_domain in DOMAINS:
    dataset = datasets.ImageFolder(
        root=f"{DATA_ROOT}\\{test_domain}",
        transform=preprocess
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    trainer = Trainer(
        clip_model=clip_model, tokenizer=clip.tokenize,
        class_names=CLASS_NAMES, device=device,
        feature_dim=512, epochs=100,
        templates=[
            "a {cls} in a {style} style",
            "a {style} style of a {cls}",
            "a photo of a {cls} with {style} like style",
        ]
    )
    trainer.load_checkpoint("checkpoints/dpstylerpp_final.pt")

    acc = trainer.evaluate(loader, clip_model)
    results[test_domain] = acc

print("\n==== Per-Domain Results (PACS) ====")
for domain, acc in results.items():
    print(f"  {domain:15s}: {acc:.2f}%")
print(f"  {'Average':15s}: {sum(results.values())/len(results):.2f}%")
print(f"\nDPStyler baseline avg: ~93.6%")