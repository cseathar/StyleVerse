""""
  1. DomainUncertaintyLoss  
  2. DualObjectiveLoss      – Improvement 3 (inter-domain var ↓, inter-class var ↑)
  3. ArcFaceLoss            –
  4. TotalLoss              – combines all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 1.  Domain Uncertainty Loss  

class DomainUncertaintyLoss(nn.Module):
    """
    Entropy maximisation over style predictions.
    Forces the style-removed feature to look equally probable under every style
    → style information has been successfully stripped.

    L_U = sum_j  p_ij * log(p_ij)   (negative entropy, to be minimised)
    """

    def forward(self,
                removed_features: torch.Tensor,   # (N, C)  after style remover
                style_features: torch.Tensor       # (K, C)  K style text features
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        removed_features : (N, C)  – R(T(P_content ∘ P_style))
        style_features   : (K, C)  – T(P_style_j) for j=1..K

        Returns
        -------
        loss : scalar
        """
        # Cosine similarities  →  (N, K)
        sim = removed_features @ style_features.T        # (N, K)
        # Softmax probabilities
        p   = F.softmax(sim, dim=-1)                     # (N, K)
        # Negative entropy (to minimise = maximise entropy)
        loss = (p * torch.log(p + 1e-8)).sum(dim=-1).mean()
        return loss



# 2.  Dual-Objective Channel Refinement Loss  (Improvement 3 – CLIPCEIL idea)

class DualObjectiveLoss(nn.Module):
    """
    Two complementary objectives on the style-removed features:

      L_intra : minimise variance of same-class features across different styles
                → features should be style-agnostic within a class

      L_inter : maximise variance of per-class mean features across classes
                → features should still be discriminative across classes

      L_dual  = L_intra  -  lambda_inter * L_inter

    This is more principled than entropy-only loss: it explicitly controls
    BOTH style suppression AND class discrimination.
    """

    def __init__(self, lambda_inter: float = 0.5):
        super().__init__()
        self.lambda_inter = lambda_inter

    def forward(self,
                features: torch.Tensor,    # (N, C)  after style remover
                class_ids: torch.Tensor    # (N,)    integer class index
                ) -> torch.Tensor:
        unique_classes = class_ids.unique()

        # ── Term 1: intra-class inter-style variance (minimise) ──────────
        intra_vars = []
        class_means = []
        for cls in unique_classes:
            mask      = class_ids == cls
            cls_feats = features[mask]             # (k, C)
            class_means.append(cls_feats.mean(dim=0))
            if cls_feats.shape[0] > 1:
                # mean per-channel variance across styles
                intra_vars.append(cls_feats.var(dim=0).mean())

        L_intra = (torch.stack(intra_vars).mean()
                   if intra_vars
                   else features.new_zeros(1).squeeze())

        # ── Term 2: inter-class variance (maximise) ───────────────────────
        if len(class_means) > 1:
            class_means_t = torch.stack(class_means, dim=0)   # (C_cls, C)
            L_inter = class_means_t.var(dim=0).mean()
        else:
            L_inter = features.new_zeros(1).squeeze()

        return L_intra - self.lambda_inter * L_inter


# 3.  ArcFace Loss  

class ArcFaceLoss(nn.Module):
    """
    Additive angular margin loss.
    Keeps same-class features close and different-class features far apart
    on the unit hypersphere.

    Reference: Deng et al., CVPR 2019
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 scale: float = 5.0, margin: float = 0.5):
        super().__init__()
        self.scale   = scale
        self.margin  = margin
        self.cos_m   = math.cos(margin)
        self.sin_m   = math.sin(margin)
        self.th      = math.cos(math.pi - margin)
        self.mm      = math.sin(math.pi - margin) * margin

        # Weight matrix: each column = class prototype
        self.weight = nn.Parameter(
            torch.empty(feature_dim, num_classes)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self,
                features: torch.Tensor,   # (N, C)  L2-normalised
                labels: torch.Tensor      # (N,)    integer class labels
                ) -> torch.Tensor:
        # Normalise weight prototypes
        W    = F.normalize(self.weight, dim=0)          # (C, num_classes)
        cos  = features @ W                             # (N, num_classes)

        # Compute cos(θ + m) for target class
        sin  = (1.0 - cos.pow(2)).clamp(0).sqrt()
        phi  = cos * self.cos_m - sin * self.sin_m

        # Numerical stability: if cos(θ) < threshold, use linear approximation
        phi  = torch.where(cos > self.th, phi, cos - self.mm)

        # One-hot mask
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        logits = (one_hot * phi + (1.0 - one_hot) * cos) * self.scale
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def get_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Plain cosine similarity logits for inference."""
        W = F.normalize(self.weight, dim=0)
        return (features @ W) * self.scale



# 4.  Total Loss  (combines all components)

class TotalLoss(nn.Module):
    """
    L_total = w_domain * L_U  +  w_dual * L_dual  +  w_cls * L_cls

    Weights are configurable; paper-inspired defaults are:
      w_domain = 1.0   (original DPStyler)
      w_dual   = 0.5   (new Improvement 3)
      w_cls    = 1.0   (original DPStyler ArcFace)
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 scale: float = 5.0, margin: float = 0.5,
                 lambda_inter: float = 0.5,
                 w_domain: float = 1.0,
                 w_dual:   float = 0.5,
                 w_cls:    float = 1.0):
        super().__init__()

        self.w_domain = w_domain
        self.w_dual   = w_dual
        self.w_cls    = w_cls

        self.domain_loss = DomainUncertaintyLoss()
        self.dual_loss   = DualObjectiveLoss(lambda_inter=lambda_inter)
        self.cls_loss    = ArcFaceLoss(feature_dim, num_classes,
                                       scale=scale, margin=margin)

    def forward(self,
                removed_features: torch.Tensor,   # (N, C)
                style_features:   torch.Tensor,   # (K, C)
                class_ids:        torch.Tensor,   # (N,)
                labels:           torch.Tensor    # (N,)  same as class_ids in training
                ) -> dict[str, torch.Tensor]:

        L_domain = self.domain_loss(removed_features, style_features)
        L_dual   = self.dual_loss(removed_features, class_ids)
        L_cls    = self.cls_loss(removed_features, labels)

        L_total  = (self.w_domain * L_domain
                    + self.w_dual   * L_dual
                    + self.w_cls    * L_cls)

        return {
            "total":  L_total,
            "domain": L_domain.detach(),
            "dual":   L_dual.detach(),
            "cls":    L_cls.detach(),
        }
