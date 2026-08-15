"""


  Improvement 1 - Gaussian Distribution Style Modelling   (style_remover.py)
  Improvement 2 - LLM-Generated Semantic Styles           (style_generator.py)
  Improvement 3 - Dual-Objective Channel Refinement Loss  (losses.py)
  Improvement 4 - ViT-B/16 or ViT-L/14 backbone          (CLIP model choice)
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.style_remover   import EnhancedStyleRemover
from models.style_generator import StyleVectorGenerator, LLMStyleGenerator
from losses.losses          import TotalLoss


class DPStylerPlusPlus(nn.Module):

    def __init__(self,
                 clip_model,
                 tokenizer,
                 class_names,
                 device,
                 feature_dim    = 512,
                 K              = 80,
                 resample_n     = 4,
                 lambda_inter   = 0.5,
                 scale          = 5.0,
                 margin         = 0.5,
                 templates      = None,
                 llm_api_key    = None,
                 llm_num_styles = 200):

        super().__init__()

        self.clip_model  = clip_model
        self.tokenizer   = tokenizer
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device      = device
        self.feature_dim = feature_dim
        self.K           = K

        self.templates = templates or [
            "a {cls} in a {style} style",
            "a {style} style of a {cls}",
            "a photo of a {cls} with {style} like style",
        ]

        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        llm_gen = LLMStyleGenerator(
            num_styles = llm_num_styles,
            api_key    = llm_api_key
        )
        self.style_generators = []
        for _ in self.templates:
            gen = StyleVectorGenerator(
                clip_model     = clip_model,
                tokenizer      = tokenizer,
                device         = device,
                feature_dim    = feature_dim,
                K              = K,
                llm_generator  = llm_gen,
                strategy_probs = (0.25, 0.25, 0.50),
            )
            self.style_generators.append(gen)

        self.removers = nn.ModuleList([
            EnhancedStyleRemover(
                feature_dim  = feature_dim,
                num_classes  = self.num_classes,
                lambda_inter = lambda_inter,
                resample_n   = resample_n,
            )
            for _ in self.templates
        ])

        self.criterions = nn.ModuleList([
            TotalLoss(
                feature_dim  = feature_dim,
                num_classes  = self.num_classes,
                scale        = scale,
                margin       = margin,
                lambda_inter = lambda_inter,
            )
            for _ in self.templates
        ])

    # ------------------------------------------------------------------
    def initialise_generators(self):
        self.style_generators[0].initialise()
        for gen in self.style_generators[1:]:
            gen._llm_styles   = self.style_generators[0]._llm_styles
            gen._seed_vectors = self.style_generators[0]._seed_vectors
            gen._llm_vectors  = self.style_generators[0]._llm_vectors

    # ------------------------------------------------------------------
    def _get_token_embedding(self):
        if hasattr(self.clip_model, 'token_embedding'):
            return self.clip_model.token_embedding
        if hasattr(self.clip_model, 'model'):
            return self.clip_model.model.token_embedding
        raise AttributeError("Cannot find token_embedding in CLIP model.")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_prompts(self, template, style_vectors):
        """
        Encode all (class x style) prompts in ONE batched encode_text call.
        This is the correct DPStyler approach — fast and accurate.

        For each class we:
          1. Build the base prompt  e.g. "a dog in a distinctive style"
          2. Tokenize it once
          3. Patch the embedding table at dummy_id with each style vector
          4. Call encode_text once per style (batched across classes)

        Returns features (M*K, D) and class_ids (M*K,).
        """
        token_emb  = self._get_token_embedding()
        clip_dtype = token_emb.weight.dtype

        # Tokenize dummy word to find its token id
        dummy_tokens = self.tokenizer(["distinctive"]).to(self.device)
        dummy_id     = dummy_tokens[0, 1].item()

        # Build all prompts for all classes
        prompts = []
        for cls_name in self.class_names:
            prompt = template.replace("{cls}", cls_name).replace(
                "{style}", "distinctive"
            )
            prompts.append(prompt)

        # Tokenize all class prompts  ->  (M, 77)
        all_tokens = self.tokenizer(prompts).to(self.device)

        all_feats = []
        all_ids   = []

        # Save original embedding for dummy_id
        original_emb = token_emb.weight.data[dummy_id].clone()

        for k in range(self.K):
            # Inject k-th style vector into embedding table
            style_norm = F.normalize(
                style_vectors[k].unsqueeze(0), dim=-1
            ).to(clip_dtype)
            emb_scale = original_emb.norm()
            token_emb.weight.data[dummy_id] = (style_norm * emb_scale).squeeze(0)

            # Encode ALL classes at once with this style  ->  (M, D)
            feats = self.clip_model.encode_text(all_tokens).float()
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats)

        # Restore original embedding
        token_emb.weight.data[dummy_id] = original_emb

        # Stack:  all_feats is list of K tensors each (M, D)
        # Rearrange to (M*K, D) grouped by class
        stacked   = torch.stack(all_feats, dim=1)      # (M, K, D)
        features  = stacked.reshape(-1, self.feature_dim)  # (M*K, D)

        # Build class ids: [0,0,...,0, 1,1,...,1, ..., M-1,...,M-1]
        # each class repeated K times
        for cls_idx in range(self.num_classes):
            all_ids.append(
                torch.full((self.K,), cls_idx,
                           dtype=torch.long, device=self.device)
            )
        class_ids = torch.cat(all_ids, dim=0)          # (M*K,)

        return features, class_ids

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_style_prompts(self, style_vectors):
        """
        Encode K style-only prompts in one batched call.
        Returns (K, D) style features for domain uncertainty loss.
        """
        token_emb  = self._get_token_embedding()
        clip_dtype = token_emb.weight.dtype

        dummy_tokens = self.tokenizer(["distinctive"]).to(self.device)
        dummy_id     = dummy_tokens[0, 1].item()

        # One style-only prompt
        style_prompt = "a distinctive style"
        tokens = self.tokenizer([style_prompt]).to(self.device)

        original_emb = token_emb.weight.data[dummy_id].clone()
        style_feats  = []

        for k in range(self.K):
            style_norm = F.normalize(
                style_vectors[k].unsqueeze(0), dim=-1
            ).to(clip_dtype)
            emb_scale = original_emb.norm()
            token_emb.weight.data[dummy_id] = (style_norm * emb_scale).squeeze(0)

            feat = self.clip_model.encode_text(tokens).float()
            feat = F.normalize(feat, dim=-1)
            style_feats.append(feat)

        token_emb.weight.data[dummy_id] = original_emb

        return torch.cat(style_feats, dim=0)            # (K, D)

    # ------------------------------------------------------------------
    def train_epoch(self, optimizers):
        """Run one training epoch across all template models."""
        self.train()
        metrics = {"total": [], "domain": [], "dual": [], "cls": []}

        for t_idx, (template, gen, remover, criterion) in enumerate(
            zip(self.templates, self.style_generators,
                self.removers, self.criterions)
        ):
            opt = optimizers[t_idx]
            opt.zero_grad()

            # Step 1: Refresh style vectors
            style_vectors = gen.refresh()

            # Step 2: Encode all (class x style) prompts
            features, class_ids = self._encode_prompts(template, style_vectors)
            features = features.detach()

            # Step 3: Gaussian update + resample  (Improvement 1)
            extra_feats, extra_ids = remover.update_gaussian_and_resample(
                features, class_ids
            )
            if extra_feats.shape[0] > 0:
                features  = torch.cat([features,  extra_feats], dim=0)
                class_ids = torch.cat([class_ids, extra_ids],   dim=0)

            features = features.detach()

            # Step 4: Style removal
            removed = remover(features)
            removed = F.normalize(removed, dim=-1)

            # Step 5: Style reference features
            style_feats = self._encode_style_prompts(style_vectors)

            # Step 6: Compute total loss  (Improvement 3 dual loss inside)
            loss_dict = criterion(
                removed_features = removed,
                style_features   = style_feats,
                class_ids        = class_ids,
                labels           = class_ids,
            )

            loss_dict["total"].backward()
            opt.step()

            for k in metrics:
                metrics[k].append(loss_dict[k].item())

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image_features):
        image_features = F.normalize(image_features, dim=-1)
        all_logits = []

        for remover, criterion in zip(self.removers, self.criterions):
            removed = remover(image_features)
            removed = F.normalize(removed, dim=-1)
            logits  = criterion.cls_loss.get_logits(removed)
            all_logits.append(logits)

        stacked = torch.stack(all_logits, dim=0)
        max_logits, _ = stacked.max(dim=0)
        return max_logits.argmax(dim=-1)

    @torch.no_grad()
    def predict_proba(self, image_features):
        image_features = F.normalize(image_features, dim=-1)
        all_logits = []

        for remover, criterion in zip(self.removers, self.criterions):
            removed = remover(image_features)
            removed = F.normalize(removed, dim=-1)
            logits  = criterion.cls_loss.get_logits(removed)
            all_logits.append(logits)

        stacked = torch.stack(all_logits, dim=0)
        max_logits, _ = stacked.max(dim=0)
        return F.softmax(max_logits, dim=-1)


# Trainer helper
class Trainer:

    def __init__(self,
                 clip_model,
                 tokenizer,
                 class_names,
                 device,
                 feature_dim    = 512,
                 epochs         = 100,
                 lr             = 0.008,
                 momentum       = 0.9,
                 K              = 80,
                 resample_n     = 4,
                 lambda_inter   = 0.5,
                 scale          = 5.0,
                 margin         = 0.5,
                 templates      = None,
                 llm_api_key    = None,
                 checkpoint_dir = "checkpoints"):

        self.device   = device
        self.epochs   = epochs
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(exist_ok=True)

        self.model = DPStylerPlusPlus(
            clip_model    = clip_model,
            tokenizer     = tokenizer,
            class_names   = class_names,
            device        = device,
            feature_dim   = feature_dim,
            K             = K,
            resample_n    = resample_n,
            lambda_inter  = lambda_inter,
            scale         = scale,
            margin        = margin,
            templates     = templates,
            llm_api_key   = llm_api_key,
        ).to(device)

        self.optimizers = [
            SGD(
                list(remover.parameters()) + list(criterion.parameters()),
                lr=lr, momentum=momentum
            )
            for remover, criterion in zip(
                self.model.removers, self.model.criterions
            )
        ]

        self.schedulers = [
            CosineAnnealingLR(opt, T_max=epochs)
            for opt in self.optimizers
        ]

    # ------------------------------------------------------------------
    def fit(self, verbose=True):
        self.model.initialise_generators()
        history = []
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            metrics = self.model.train_epoch(self.optimizers)
            for sched in self.schedulers:
                sched.step()

            history.append({"epoch": epoch, **metrics})

            if verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:3d}/{self.epochs}  "
                    f"loss={metrics['total']:.4f}  "
                    f"domain={metrics['domain']:.4f}  "
                    f"dual={metrics['dual']:.4f}  "
                    f"cls={metrics['cls']:.4f}  "
                    f"[{elapsed:.1f}s elapsed]"
                )

        self.save_checkpoint("final")
        return history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, clip_model):
        """
        NOTE for Windows: always use num_workers=0 in your DataLoader.
        """
        self.model.eval()
        correct = total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            img_feats = clip_model.encode_image(images).float()
            img_feats = F.normalize(img_feats, dim=-1)

            preds   = self.model.predict(img_feats)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]

        acc = correct / total * 100.0
        print(f"Top-1 Accuracy: {acc:.2f}%  ({correct}/{total})")
        return acc

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag="best"):
        path = self.ckpt_dir / f"dpstylerpp_{tag}.pt"
        torch.save({
            "removers":   [r.state_dict() for r in self.model.removers],
            "criterions": [c.state_dict() for c in self.model.criterions],
        }, path)
        print(f"[Trainer] Checkpoint saved -> {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        for r, sd in zip(self.model.removers,   ckpt["removers"]):
            r.load_state_dict(sd)
        for c, sd in zip(self.model.criterions, ckpt["criterions"]):
            c.load_state_dict(sd)
        print(f"[Trainer] Checkpoint loaded <- {path}")
