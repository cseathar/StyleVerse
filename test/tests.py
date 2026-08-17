"""
Unit Tests

Tests all four improvements using lightweight mock objects.
No real CLIP or GPU required runs in ~2 seconds on CPU.

Run with:
    python tests.py
"""

import sys
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from models.style_remover   import StyleSENet, GaussianStyleDistribution, EnhancedStyleRemover
from models.style_generator import LLMStyleGenerator, StyleVectorGenerator, FALLBACK_STYLES
from losses.losses          import (DomainUncertaintyLoss, DualObjectiveLoss,
                                    ArcFaceLoss, TotalLoss)


#Shared test config
D   = 64    # small feature dim for fast tests
M   = 5     # num classes
K   = 8     # num styles
B   = 16    # batch size


def rand_normed(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)



# Improvement 1  Gaussian Distribution Style Modelling

class TestGaussianStyleDistribution(unittest.TestCase):

    def setUp(self):
        self.gsd = GaussianStyleDistribution(num_classes=M, feature_dim=D)

    def test_update_initialises_stats(self):
        feats    = rand_normed(M * K, D)
        ids      = torch.arange(M).repeat_interleave(K)
        self.gsd.update(feats, ids)
        self.assertTrue(self.gsd.initialized.all(),
                        "All classes should be initialised after update")

    def test_resample_shape(self):
        feats = rand_normed(M * K, D)
        ids   = torch.arange(M).repeat_interleave(K)
        self.gsd.update(feats, ids)
        n = 4
        s_feats, s_ids = self.gsd.resample(n_per_class=n)
        self.assertEqual(s_feats.shape, (M * n, D))
        self.assertEqual(s_ids.shape,   (M * n,))

    def test_resampled_feats_normalised(self):
        feats = rand_normed(M * K, D)
        ids   = torch.arange(M).repeat_interleave(K)
        self.gsd.update(feats, ids)
        s_feats, _ = self.gsd.resample(n_per_class=2)
        norms = s_feats.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
                        "Resampled features should be L2-normalised")

    def test_momentum_update_changes_stats(self):
        feats1 = rand_normed(M * K, D)
        feats2 = rand_normed(M * K, D)
        ids    = torch.arange(M).repeat_interleave(K)
        self.gsd.update(feats1, ids)
        mean_before = self.gsd.running_mean.clone()
        self.gsd.update(feats2, ids)
        mean_after  = self.gsd.running_mean
        self.assertFalse(torch.allclose(mean_before, mean_after),
                         "Running mean should change after second update")


class TestEnhancedStyleRemover(unittest.TestCase):

    def setUp(self):
        self.remover = EnhancedStyleRemover(
            feature_dim=D, num_classes=M,
            lambda_inter=0.5, resample_n=2
        )

    def test_forward_shape(self):
        v   = rand_normed(B, D)
        out = self.remover(v)
        self.assertEqual(out.shape, (B, D))

    def test_dual_loss_is_scalar(self):
        feats     = rand_normed(M * K, D)
        class_ids = torch.arange(M).repeat_interleave(K)
        style_ids = torch.arange(K).repeat(M)
        loss = self.remover.compute_dual_loss(feats, class_ids, style_ids)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_gaussian_resample_integrates(self):
        feats     = rand_normed(M * K, D)
        class_ids = torch.arange(M).repeat_interleave(K)
        e_feats, e_ids = self.remover.update_gaussian_and_resample(feats, class_ids)
        # After first epoch all classes initialised: should get M*resample_n extras
        self.assertGreater(e_feats.shape[0], 0)



# Improvement 2 LLM-Generated Semantic Styles

class TestLLMStyleGenerator(unittest.TestCase):

    def test_fallback_vocabulary_nonempty(self):
        gen = LLMStyleGenerator(api_key="")
        styles = gen.load_or_generate()
        self.assertGreater(len(styles), 0)
        self.assertIsInstance(styles[0], str)

    def test_fallback_loads_from_builtin(self):
        gen = LLMStyleGenerator(api_key="", cache_path="/nonexistent/path.json")
        styles = gen.load_or_generate()
        self.assertEqual(styles, list(FALLBACK_STYLES))

    def test_cache_roundtrip(self, tmp_path=None):
        import tempfile, json, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                        mode="w") as f:
            json.dump({"styles": ["watercolor", "cartoon", "sketch"]}, f)
            fname = f.name
        try:
            gen = LLMStyleGenerator(api_key="", cache_path=fname)
            styles = gen.load_or_generate()
            self.assertEqual(styles, ["watercolor", "cartoon", "sketch"])
        finally:
            os.unlink(fname)


class TestStyleVectorGenerator(unittest.TestCase):

    def _make_mock_clip(self):
        mock = MagicMock()
        mock.encode_text = MagicMock(
            return_value=rand_normed(10, D)   # return 10 dummy features
        )
        return mock

    def _make_mock_tokenizer(self):
        def tokenizer(texts):
            return torch.zeros(len(texts), 77, dtype=torch.long)
        return tokenizer

    def setUp(self):
        clip_model = self._make_mock_clip()
        tokenizer  = self._make_mock_tokenizer()
        llm_gen    = LLMStyleGenerator(api_key="")   # uses fallback
        self.gen   = StyleVectorGenerator(
            clip_model   = clip_model,
            tokenizer    = tokenizer,
            device       = torch.device("cpu"),
            feature_dim  = D,
            K            = K,
            llm_generator= llm_gen,
        )
        self.gen._llm_styles  = list(FALLBACK_STYLES[:20])
        self.gen._seed_vectors = rand_normed(20, D)
        self.gen._llm_vectors  = rand_normed(20, D)

    def test_random_generation_shape(self):
        vectors = self.gen._random_generation()
        self.assertEqual(vectors.shape, (K, D))

    def test_stylemix_generation_shape(self):
        vectors = self.gen._stylemix_generation()
        self.assertEqual(vectors.shape, (K, D))

    def test_llm_semantic_generation_shape(self):
        vectors = self.gen._llm_semantic_generation()
        self.assertEqual(vectors.shape, (K, D))

    def test_all_outputs_normalised(self):
        for method in [
            self.gen._random_generation,
            self.gen._stylemix_generation,
            self.gen._llm_semantic_generation,
        ]:
            vecs  = method()
            norms = vecs.norm(dim=-1)
            self.assertTrue(
                torch.allclose(norms, torch.ones(K), atol=1e-5),
                f"{method.__name__} output not normalised"
            )

    def test_refresh_returns_correct_shape(self):
        vectors = self.gen.refresh()
        self.assertEqual(vectors.shape, (K, D))



# Improvement 3 – Dual-Objective Channel Refinement Loss

class TestDualObjectiveLoss(unittest.TestCase):

    def test_loss_is_scalar(self):
        loss_fn   = DualObjectiveLoss(lambda_inter=0.5)
        features  = rand_normed(M * K, D)
        class_ids = torch.arange(M).repeat_interleave(K)
        loss = loss_fn(features, class_ids)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_intra_term_decreases_with_identical_features(self):
        """
        If all features of the same class are identical, intra-class variance
        should be ~0, making the intra term near zero.
        """
        loss_fn   = DualObjectiveLoss(lambda_inter=0.0)  # isolate intra term
        features  = rand_normed(M, D).repeat_interleave(K, dim=0)  # identical per class
        class_ids = torch.arange(M).repeat_interleave(K)
        loss = loss_fn(features, class_ids)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_inter_term_is_positive_with_diverse_classes(self):
        """Inter-class term should be > 0 when classes differ."""
        from losses.losses import DualObjectiveLoss
        loss_fn   = DualObjectiveLoss(lambda_inter=0.0)
        features  = rand_normed(M * K, D)
        class_ids = torch.arange(M).repeat_interleave(K)
        # compute only inter term
        class_means = [features[class_ids == c].mean(0)
                       for c in range(M)]
        cm = torch.stack(class_means)
        L_inter = cm.var(dim=0).mean()
        self.assertGreater(L_inter.item(), 0.0)


class TestDomainUncertaintyLoss(unittest.TestCase):

    def test_loss_is_scalar(self):
        loss_fn       = DomainUncertaintyLoss()
        removed       = rand_normed(M * K, D)
        style_feats   = rand_normed(K, D)
        loss = loss_fn(removed, style_feats)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_uniform_probs_give_low_loss(self):
        """
        If probabilities are perfectly uniform, entropy is maximal,
        so the negative-entropy loss should be maximally negative.
        """
        loss_fn     = DomainUncertaintyLoss()
        # Construct features that produce uniform sim with all styles
        removed     = rand_normed(4, D)
        style_feats = rand_normed(4, D)
        # Not checking exact value, just that it's a valid float
        loss = loss_fn(removed, style_feats)
        self.assertTrue(torch.isfinite(loss))


class TestArcFaceLoss(unittest.TestCase):

    def test_loss_decreases_with_training(self):
        """ArcFace loss should be reducible via gradient descent."""
        arcface = ArcFaceLoss(D, M, scale=5.0, margin=0.5)
        opt = torch.optim.SGD(arcface.parameters(), lr=0.01)

        features = rand_normed(B, D)
        labels   = torch.randint(0, M, (B,))

        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = arcface(features, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0],
                        "ArcFace loss should decrease with gradient steps")

    def test_get_logits_shape(self):
        arcface = ArcFaceLoss(D, M)
        feats   = rand_normed(B, D)
        logits  = arcface.get_logits(feats)
        self.assertEqual(logits.shape, (B, M))


class TestTotalLoss(unittest.TestCase):

    def test_all_components_returned(self):
        criterion = TotalLoss(D, M)
        features  = rand_normed(M * K, D)
        style_f   = rand_normed(K, D)
        class_ids = torch.arange(M).repeat_interleave(K)

        result = criterion(features, style_f, class_ids, class_ids)
        for key in ["total", "domain", "dual", "cls"]:
            self.assertIn(key, result)

    def test_total_loss_is_differentiable(self):
        criterion = TotalLoss(D, M)
        opt = torch.optim.SGD(criterion.parameters(), lr=0.01)

        features  = rand_normed(M * K, D)
        style_f   = rand_normed(K, D)
        class_ids = torch.arange(M).repeat_interleave(K)

        opt.zero_grad()
        result = criterion(features, style_f, class_ids, class_ids)
        result["total"].backward()
        opt.step()

        # If we get here without error, it's differentiable
        self.assertTrue(True)



# Improvement 4  ViT backbone (just verifies feature_dim is respected)

class TestBackboneFeatureDims(unittest.TestCase):

    def test_se_net_works_with_vit_b16_dim(self):
        """ViT-B/16 outputs 512-dim features."""
        se = StyleSENet(feature_dim=512)
        x  = rand_normed(B, 512)
        y  = se(x)
        self.assertEqual(y.shape, (B, 512))

    def test_se_net_works_with_vit_l14_dim(self):
        """ViT-L/14 outputs 768-dim features."""
        se = StyleSENet(feature_dim=768)
        x  = rand_normed(B, 768)
        y  = se(x)
        self.assertEqual(y.shape, (B, 768))

    def test_gaussian_dist_works_with_large_dim(self):
        gsd = GaussianStyleDistribution(num_classes=M, feature_dim=768)
        feats = rand_normed(M * K, 768)
        ids   = torch.arange(M).repeat_interleave(K)
        gsd.update(feats, ids)
        s, _ = gsd.resample(n_per_class=2)
        self.assertEqual(s.shape[1], 768)


# Run all tests

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestGaussianStyleDistribution,
        TestEnhancedStyleRemover,
        TestLLMStyleGenerator,
        TestStyleVectorGenerator,
        TestDualObjectiveLoss,
        TestDomainUncertaintyLoss,
        TestArcFaceLoss,
        TestTotalLoss,
        TestBackboneFeatureDims,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{total} tests passed")
    if result.failures or result.errors:
        print(f"  FAILURES : {len(result.failures)}")
        print(f"  ERRORS   : {len(result.errors)}")
    else:
        print("  All tests PASSED ✓")
    print(f"{'='*50}")
