

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1.  Style-SE Net  

class StyleSENet(nn.Module):
    """
    Squeeze-and-Excitation style remover.
    Learns to suppress style-carrying channels and highlight semantic ones.
    """

    def __init__(self, feature_dim: int, reduction_ratio: int = 4):
        super().__init__()
        hidden = max(feature_dim // reduction_ratio, 16)
        self.fc1 = nn.Linear(feature_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, feature_dim, bias=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v : (B, C)
        a = torch.sigmoid(self.fc2(F.relu(self.fc1(v))))   # (B, C)
        return a * v + v                                    # residual



# 2.  Gaussian Style Distribution  (Improvement 1)

class GaussianStyleDistribution(nn.Module):
    """
    For each class j, maintains a running Gaussian N(mu_j, sigma_j^2) over
    style-augmented features.  During training we RESAMPLE from this
    distribution to produce extra synthetic features, dramatically increasing
    effective style coverage without any additional text prompts.

    Usage
    -----
    gsd = GaussianStyleDistribution(num_classes, feature_dim)

    # after computing text features per (class, style):
    gsd.update(features, class_ids)          # update running stats
    extra_feats, extra_ids = gsd.resample(n_extra_per_class)  # get extras
    """

    def __init__(self, num_classes: int, feature_dim: int,
                 momentum: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum

        # Running statistics (not learned parameters – just buffers)
        self.register_buffer("running_mean",
                             torch.zeros(num_classes, feature_dim))
        self.register_buffer("running_var",
                             torch.ones(num_classes, feature_dim))
        self.register_buffer("initialized",
                             torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor, class_ids: torch.Tensor):
        """
        Update running Gaussian statistics for each class.

        Parameters
        ----------
        features  : (N, C)  – L2-normalised text features
        class_ids : (N,)    – integer class index for each feature
        """
        for cls in class_ids.unique():
            mask = class_ids == cls
            cls_feats = features[mask]                   # (k, C)
            batch_mean = cls_feats.mean(dim=0)
            batch_var  = cls_feats.var(dim=0) + 1e-6

            if not self.initialized[cls]:
                self.running_mean[cls] = batch_mean
                self.running_var[cls]  = batch_var
                self.initialized[cls]  = True
            else:
                m = self.momentum
                self.running_mean[cls] = (1 - m) * self.running_mean[cls] + m * batch_mean
                self.running_var[cls]  = (1 - m) * self.running_var[cls]  + m * batch_var

    def resample(self, n_per_class: int = 4,
                 only_initialized: bool = True
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n_per_class synthetic features for every (initialised) class.

        Returns
        -------
        feats     : (M, C)  M = num_classes * n_per_class
        class_ids : (M,)
        """
        device = self.running_mean.device
        feats_list, ids_list = [], []

        classes = (self.initialized.nonzero(as_tuple=True)[0]
                   if only_initialized
                   else torch.arange(self.num_classes, device=device))

        for cls in classes:
            mu    = self.running_mean[cls]          # (C,)
            sigma = self.running_var[cls].sqrt()    # (C,)
            eps   = torch.randn(n_per_class, self.feature_dim, device=device)
            samples = mu.unsqueeze(0) + eps * sigma.unsqueeze(0)  # (n, C)
            # Re-normalise to unit sphere (CLIP lives on hypersphere)
            samples = F.normalize(samples, dim=-1)
            feats_list.append(samples)
            ids_list.append(torch.full((n_per_class,), cls.item(),
                                       dtype=torch.long, device=device))

        if not feats_list:
            return (torch.empty(0, self.feature_dim, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        return torch.cat(feats_list, dim=0), torch.cat(ids_list, dim=0)



# 3.  Enhanced Style Remover  (Improvement 1 + 3 combined)

class EnhancedStyleRemover(nn.Module):
    """
    Wraps StyleSENet and adds:
      • per-class Gaussian distribution tracking & resampling   (Improvement 1)
      • dual-objective channel refinement loss computation       (Improvement 3)

    The dual-objective loss has two terms:
      L_intra  = mean inter-style variance for SAME class   (minimise → style-agnostic)
      L_inter  = mean inter-class variance across classes   (maximise → class-discriminative)

    Combined: L_dual = L_intra - lambda_inter * L_inter
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 reduction_ratio: int = 4,
                 lambda_inter: float = 0.5,
                 resample_n: int = 4,
                 gaussian_momentum: float = 0.1):
        super().__init__()

        self.feature_dim   = feature_dim
        self.num_classes   = num_classes
        self.lambda_inter  = lambda_inter
        self.resample_n    = resample_n

        # Core SE-Net remover
        self.se_net = StyleSENet(feature_dim, reduction_ratio)

        # Gaussian distribution tracker
        self.gaussian = GaussianStyleDistribution(
            num_classes, feature_dim, momentum=gaussian_momentum
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Standard forward: just the SE-Net pass."""
        return self.se_net(v)

    def compute_dual_loss(self,
                          features: torch.Tensor,
                          class_ids: torch.Tensor,
                          style_ids: torch.Tensor
                          ) -> torch.Tensor:
        """
        Compute the dual-objective channel refinement loss.

        Parameters
        ----------
        features  : (N, C)  AFTER style remover  (R(v))
        class_ids : (N,)    which class each sample belongs to
        style_ids : (N,)    which style each sample was generated with

        Returns
        -------
        loss : scalar tensor
        """
        # ---- Term 1: intra-class inter-style variance (minimise) 
        # For each class, collect all style variants and compute variance
        intra_var_list = []
        for cls in class_ids.unique():
            cls_mask  = class_ids == cls
            cls_feats = features[cls_mask]          # (k, C)  k = num styles
            if cls_feats.shape[0] > 1:
                var = cls_feats.var(dim=0).mean()   # scalar
                intra_var_list.append(var)

        L_intra = (torch.stack(intra_var_list).mean()
                   if intra_var_list else features.new_zeros(1).squeeze())

        # ---- Term 2: inter-class variance (maximise) 
        # Compute per-class mean features, then their variance across classes
        class_means = []
        for cls in class_ids.unique():
            cls_mask = class_ids == cls
            class_means.append(features[cls_mask].mean(dim=0))

        if len(class_means) > 1:
            class_means_t = torch.stack(class_means, dim=0)   # (num_cls, C)
            L_inter = class_means_t.var(dim=0).mean()
        else:
            L_inter = features.new_zeros(1).squeeze()

        # ---- Combined dual loss (minimise intra, maximise inter)
        loss = L_intra - self.lambda_inter * L_inter
        return loss

    def update_gaussian_and_resample(
            self,
            features: torch.Tensor,
            class_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update Gaussian stats, then return resampled synthetic features.
        Call this BEFORE the SE-Net forward pass so resampled features
        also flow through the remover.
        """
        self.gaussian.update(features.detach(), class_ids)
        return self.gaussian.resample(n_per_class=self.resample_n)
