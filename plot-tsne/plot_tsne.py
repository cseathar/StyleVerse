

import argparse
import os
import sys
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, str(Path(__file__).parent))


#Color palettes
DOMAIN_COLORS = [
    '#3B6FCA', '#E85D04', '#1D9E75', '#9333EA',
    '#DC2626', '#0891B2', '#65A30D', '#D97706',
]

CLASS_COLORS = [
    '#3B6FCA','#E85D04','#1D9E75','#9333EA','#DC2626',
    '#0891B2','#65A30D','#D97706','#BE185D','#059669',
    '#7C3AED','#B45309','#0F766E','#C2410C','#1D4ED8',
    '#15803D','#7E22CE','#B91C1C','#0369A1','#4D7C0F',
]

DATASET_DOMAINS = {
    'pacs':       ['art_painting', 'cartoon', 'photo', 'sketch'],
    'vlcs':       ['CALTECH', 'LABELME', 'SUN', 'VOC'],
    'officehome': ['Art', 'Clipart', 'Product', 'Real_World'],
    'domainnet':  ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
}


#  Safe image loader
def safe_loader(path):
    try:
        img = Image.open(path)
        img.load()
        return img.convert('RGB')
    except Exception:
        return Image.new('RGB', (224, 224), (128, 128, 128))


# Feature extraction
def extract_features(args, clip_model, preprocess, removers, device):
    """
    Extract raw CLIP features and style-removed features from dataset.
    Returns dicts with 'raw_feats', 'removed_feats', 'domain_labels', 'class_labels'.
    """
    domains   = DATASET_DOMAINS[args.dataset]
    all_raw   = []
    all_rem   = []
    all_dom   = []
    all_cls   = []

    # Get class mapping from first domain
    first_domain_path = os.path.join(args.data_root, domains[0])
    from torchvision import datasets
    ref_ds     = datasets.ImageFolder(first_domain_path)
    class_list = ref_ds.classes
    num_classes = len(class_list)
    print(f"[tsne] Classes: {num_classes}  |  Domains: {len(domains)}")

    for dom_idx, domain in enumerate(domains):
        dom_path = os.path.join(args.data_root, domain)
        if not os.path.exists(dom_path):
            print(f"[tsne] Skipping missing domain: {dom_path}")
            continue

        ds = datasets.ImageFolder(
            dom_path,
            transform=preprocess,
            loader=safe_loader
        )

        # Sample n_samples evenly across classes
        per_class = max(1, args.n_samples // num_classes)
        sampled_indices = []
        class_indices = {c: [] for c in range(num_classes)}
        for i, (_, label) in enumerate(ds.samples):
            class_indices[label].append(i)
        for c in range(num_classes):
            idxs = class_indices[c]
            if idxs:
                chosen = random.sample(idxs, min(per_class, len(idxs)))
                sampled_indices.extend(chosen)

        print(f"[tsne] {domain}: {len(sampled_indices)} samples")

        from torch.utils.data import DataLoader, Subset
        subset = Subset(ds, sampled_indices)
        loader = DataLoader(subset, batch_size=64,
                            shuffle=False, num_workers=0)

        domain_raw = []
        domain_rem = []
        domain_cls = []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)

                # Raw CLIP features
                raw = clip_model.encode_image(imgs).float()
                raw = F.normalize(raw, dim=-1)

                # Style-removed features (ensemble mean)
                rems = []
                for remover in removers:
                    rem = F.normalize(remover(raw), dim=-1)
                    rems.append(rem)
                removed = torch.stack(rems, dim=0).mean(0)
                removed = F.normalize(removed, dim=-1)

                domain_raw.append(raw.cpu())
                domain_rem.append(removed.cpu())
                domain_cls.append(labels.cpu())

        domain_raw = torch.cat(domain_raw, dim=0).numpy()
        domain_rem = torch.cat(domain_rem, dim=0).numpy()
        domain_cls = torch.cat(domain_cls, dim=0).numpy()

        all_raw.append(domain_raw)
        all_rem.append(domain_rem)
        all_dom.append(np.full(len(domain_raw), dom_idx))
        all_cls.append(domain_cls)

    raw_feats     = np.concatenate(all_raw,  axis=0)
    removed_feats = np.concatenate(all_rem,  axis=0)
    domain_labels = np.concatenate(all_dom,  axis=0)
    class_labels  = np.concatenate(all_cls,  axis=0)

    print(f"[tsne] Total samples: {len(raw_feats)}")
    return raw_feats, removed_feats, domain_labels, class_labels, domains, class_list


# t-SNE computation
def run_tsne(features, perplexity=30, n_iter=1000, random_state=42):
    print(f"[tsne] Running t-SNE on {features.shape}...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, features.shape[0] // 4),
        max_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
        verbose=0
    )
    return tsne.fit_transform(features)


# Plotting
def plot_tsne(raw_emb, rem_emb, domain_labels, class_labels,
              domains, class_list, dataset_name, output_path):

    n_domains = len(domains)
    n_classes = len(class_list)

    # Limit class colors
    cls_colors = plt.cm.tab20(
        np.linspace(0, 1, n_classes)
    )
    dom_colors = plt.cm.Set2(
        np.linspace(0, 1, n_domains)
    )

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor('#FAFBFC')

    gs = GridSpec(1, 4, figure=fig, wspace=0.08,
                  left=0.04, right=0.97, top=0.88, bottom=0.12)

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    titles = [
        'Raw CLIP Features\n(by Domain)',
        'Style-Removed Features\n(by Domain)',
        'Raw CLIP Features\n(by Class)',
        'Style-Removed Features\n(by Class)',
    ]

    def scatter_plot(ax, emb, labels, colors, title, alpha=0.6, s=18):
        unique = np.unique(labels)
        for i, lbl in enumerate(unique):
            mask = labels == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=colors[lbl], s=s, alpha=alpha,
                       edgecolors='none', zorder=3)

        # Clean axis
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#F8F9FA')
        for sp in ax.spines.values():
            sp.set_color('#DDDDDD')
            sp.set_linewidth(0.8)
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color='#1F2937', pad=10, linespacing=1.4)

    # Panel 1: Raw — by domain
    scatter_plot(axes[0], raw_emb, domain_labels, dom_colors,
                 titles[0], alpha=0.65, s=20)

    # Panel 2: Removed — by domain
    scatter_plot(axes[1], rem_emb, domain_labels, dom_colors,
                 titles[1], alpha=0.65, s=20)

    # Panel 3: Raw — by class
    scatter_plot(axes[2], raw_emb, class_labels, cls_colors,
                 titles[2], alpha=0.65, s=20)

    # Panel 4: Removed — by class
    scatter_plot(axes[3], rem_emb, class_labels, cls_colors,
                 titles[3], alpha=0.65, s=20)

    # Add annotation arrows showing improvement
    # Arrow between panel 1 and 2 (top)
    fig.text(0.5, 0.94,
             f'StyleVerse t-SNE Visualization — {dataset_name.upper()}',
             ha='center', va='center', fontsize=15, fontweight='bold',
             color='#111827')
    fig.text(0.5, 0.905,
             'Left pair: domain coloring (mixed = domain-invariant ✓) '
             '|  Right pair: class coloring (separated = discriminative ✓)',
             ha='center', va='center', fontsize=9.5, color='#6B7280',
             fontstyle='italic')

    # Domain legend
    dom_patches = [
        mpatches.Patch(facecolor=dom_colors[i], label=domains[i],
                       edgecolor='none')
        for i in range(n_domains)
    ]
    leg1 = axes[0].legend(
        handles=dom_patches,
        loc='lower left',
        fontsize=7.5,
        framealpha=0.9,
        edgecolor='#DDDDDD',
        fancybox=True,
        title='Domains',
        title_fontsize=8,
    )
    axes[1].legend(
        handles=dom_patches,
        loc='lower left',
        fontsize=7.5,
        framealpha=0.9,
        edgecolor='#DDDDDD',
        fancybox=True,
        title='Domains',
        title_fontsize=8,
    )

    # Class legend (show max 10 classes to avoid clutter)
    max_show = min(15, n_classes)
    cls_patches = [
        mpatches.Patch(facecolor=cls_colors[i],
                       label=class_list[i].replace('_', ' '),
                       edgecolor='none')
        for i in range(max_show)
    ]
    if n_classes > 10:
        cls_patches.append(mpatches.Patch(
            facecolor='#CCCCCC', label=f'+ {n_classes-10} more',
            edgecolor='none'))

    for ax_idx in [2, 3]:
        axes[ax_idx].legend(
            handles=cls_patches,
            loc='lower left',
            fontsize=6.5,
            framealpha=0.9,
            edgecolor='#DDDDDD',
            fancybox=True,
            title='Classes',
            title_fontsize=7.5,
            ncol=1 if n_classes <= 10 else 2,
        )

    #  Improvement annotations 
    for ax, note, color in [
        (axes[1], '← More mixed\n(domain invariant)', '#1D9E75'),
        (axes[3], '← Better separated\n(discriminative)', '#3B6FCA'),
    ]:
        ax.text(0.98, 0.02, note,
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=8, color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor=color,
                          linewidth=0.8, alpha=0.9))

    plt.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='#FAFBFC')
    plt.close()
    print(f"[tsne] Saved -> {output_path}")


#  Main
def get_args():
    p = argparse.ArgumentParser(description='StyleVerse t-SNE Visualization')
    p.add_argument('--dataset',    type=str, default='pacs',
                   choices=['pacs','vlcs','officehome','domainnet'])
    p.add_argument('--data_root',  type=str, default='./data')
    p.add_argument('--checkpoint', type=str,
                   default='checkpoints/dpstylerpp_finaldomainnet-sfr.pt')
    p.add_argument('--backbone',   type=str, default='ViT-B/16')
    p.add_argument('--n_samples',  type=int, default=200,
                   help='Samples per domain (total = n_samples × n_domains)')
    p.add_argument('--perplexity', type=int, default=30)
    p.add_argument('--n_iter',     type=int, default=1000)
    p.add_argument('--output',     type=str, default='tsne_output.png')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[tsne] Device: {device}")

    # Load CLIP
    try:
        import clip
    except ImportError:
        sys.exit("pip install git+https://github.com/openai/CLIP.git")

    print(f"[tsne] Loading CLIP {args.backbone}...")
    clip_model, preprocess = clip.load(args.backbone, device=device)
    clip_model.eval()

    # Load checkpoint
    print(f"[tsne] Loading checkpoint: {args.checkpoint}")
    ckpt        = torch.load(args.checkpoint, map_location=device)
    feature_dim = ckpt['feature_dim']
    class_names = ckpt['class_names']
    num_classes  = len(class_names)

    from models.style_remover import EnhancedStyleRemover
    from losses.losses        import TotalLoss

    num_templates = 3
    removers = nn.ModuleList([
        EnhancedStyleRemover(feature_dim, num_classes)
        for _ in range(num_templates)
    ])
    for r, sd in zip(removers, ckpt['removers']):
        r.load_state_dict(sd)
    removers = removers.to(device).eval()

    print(f"[tsne] Models loaded. Extracting features...")

    # Extract features
    raw_feats, rem_feats, dom_labels, cls_labels, domains, class_list = \
        extract_features(args, clip_model, preprocess, removers, device)

    # Run t-SNE on both raw and removed features
    print(f"[tsne] Computing t-SNE for raw features...")
    raw_emb = run_tsne(raw_feats, args.perplexity, args.n_iter, args.seed)

    print(f"[tsne] Computing t-SNE for style-removed features...")
    rem_emb = run_tsne(rem_feats, args.perplexity, args.n_iter, args.seed + 1)
    #Quantitative t-SNE Metrics

    from sklearn.metrics import davies_bouldin_score, silhouette_score
    db_raw = davies_bouldin_score(raw_feats, dom_labels)
    db_rem =davies_bouldin_score(rem_feats, dom_labels)
    print(f"Domain DB Score: raw={db_raw:.3f} -> removed={db_rem:.3f} (lower=more mixed)")
    sil_raw = silhouette_score(raw_feats, cls_labels, sample_size=1000)
    sil_rem=silhouette_score(rem_feats, cls_labels, sample_size=1000)
    print(f"Class Silhouette: raw={sil_raw:.3f} -> removed={sil_rem:3f} (higher=more separated)")

    # Plot
    plot_tsne(raw_emb, rem_emb, dom_labels, cls_labels,
              domains, class_list, args.dataset, args.output)

    print(f"\n[tsne] Done! Open: {args.output}")


if __name__ == '__main__':
    main()
