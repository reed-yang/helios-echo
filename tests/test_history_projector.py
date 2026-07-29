import unittest

import torch

from helios.utils.utils_base import HistoryProjector


def full_masks(tiers):
    """Frame masks that mark every frame valid."""
    return [torch.ones(tier.shape[0], tier.shape[2], dtype=torch.bool) for tier in tiers]


def history_tiers(zero_frames=0, generator=None, scale=1.0, shift=0.0):
    """Three history tiers shaped like a real section: long 16, mid 2, short 1
    (the x0 anchor is stripped by the caller in the pipeline). The first
    `zero_frames` frames of the long tier are zero padding, as in the first
    sections of a rollout."""
    tiers = []
    for frames in (16, 2, 1):
        tier = torch.randn(1, 16, frames, 6, 8, generator=generator) * scale + shift
        tiers.append(tier)
    if zero_frames:
        tiers[0][:, :, :zero_frames] = 0.0
    return tiers


class HistoryProjectorTest(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(0)

    def test_none_mode_is_disabled_and_returns_inputs(self):
        projector = HistoryProjector(mode="none")
        self.assertFalse(projector.enabled)
        tiers = history_tiers(generator=self.generator)
        out = projector(tiers)
        for original, projected in zip(tiers, out):
            self.assertIs(original, projected)

    def test_zero_alpha_is_a_no_op(self):
        projector = HistoryProjector(mode="quantize", step=0.5, alpha=0.0)
        self.assertFalse(projector.enabled)
        tiers = history_tiers(generator=self.generator)
        for original, projected in zip(tiers, projector(tiers)):
            torch.testing.assert_close(original, projected)

    def test_quantize_snaps_valid_frames_to_the_grid(self):
        step = 0.25
        projector = HistoryProjector(mode="quantize", step=step)
        tiers = history_tiers(generator=self.generator)
        out = projector(tiers)
        for projected in out:
            residual = projected / step - torch.round(projected / step)
            self.assertLess(residual.abs().max().item(), 1e-5)
        self.assertEqual(projector.stats()["num_applied"], 1)

    def test_quantize_does_not_remove_a_sub_step_bias(self):
        # Uniform scalar quantization is not a bias remover. For values spread
        # over many grid cells, a bias b < step / 2 makes about b / step of the
        # pixels jump a full step, so the mean shift survives the projection.
        # This is why quantize is a control arm and not the primary candidate
        # against a drift that shows up as a per-channel statistics shift.
        step = 0.5
        bias = 0.1
        projector = HistoryProjector(mode="quantize", step=step)
        tiers = history_tiers(generator=self.generator)
        base = projector(tiers)
        after = projector([tier + bias for tier in tiers])
        base_mean = HistoryProjector._channel_values(base, full_masks(base)).mean().item()
        after_mean = HistoryProjector._channel_values(after, full_masks(after)).mean().item()
        self.assertAlmostEqual(after_mean - base_mean, bias, delta=0.02)

    def test_zero_padding_passes_through_untouched(self):
        projector = HistoryProjector(mode="quantize", step=0.5)
        tiers = history_tiers(zero_frames=10, generator=self.generator)
        out = projector(tiers)
        self.assertEqual(out[0][:, :, :10].abs().max().item(), 0.0)
        residual = out[0][:, :, 10:] / 0.5 - torch.round(out[0][:, :, 10:] / 0.5)
        self.assertLess(residual.abs().max().item(), 1e-5)

    def test_all_zero_history_is_left_alone(self):
        projector = HistoryProjector(mode="quantize", step=0.5)
        tiers = [torch.zeros(1, 16, frames, 6, 8) for frames in (16, 2, 1)]
        out = projector(tiers)
        for projected in out:
            self.assertEqual(projected.abs().max().item(), 0.0)
        self.assertEqual(projector.stats()["num_applied"], 0)

    def test_alpha_interpolates_between_identity_and_projection(self):
        step = 0.5
        tiers = history_tiers(generator=self.generator)
        full = HistoryProjector(mode="quantize", step=step)(tiers)
        half = HistoryProjector(mode="quantize", step=step, alpha=0.5)(tiers)
        for original, projected, blended in zip(tiers, full, half):
            torch.testing.assert_close(blended, 0.5 * (original + projected))

    def test_codebook_maps_every_pixel_onto_a_centroid(self):
        codebook = torch.randn(8, 16, generator=self.generator)
        projector = HistoryProjector(mode="codebook", codebook=codebook)
        tiers = history_tiers(generator=self.generator)
        for projected in projector(tiers):
            pixels = projected.permute(0, 2, 3, 4, 1).reshape(-1, 16)
            # Exact row membership: cdist would only prove it to its own
            # matmul precision, which is ~1e-3 at this magnitude.
            unique = torch.unique(pixels, dim=0)
            on_codebook = (unique.unsqueeze(1) == codebook.unsqueeze(0)).all(dim=2).any(dim=1)
            self.assertTrue(bool(on_codebook.all()))
            self.assertLessEqual(unique.shape[0], codebook.shape[0])

    def test_codebook_never_replaces_zero_padding(self):
        # A centroid is a real latent vector, so quantizing zero padding would
        # inject content into a history window that is still empty.
        codebook = torch.randn(8, 16, generator=self.generator) + 3.0
        projector = HistoryProjector(mode="codebook", codebook=codebook)
        tiers = history_tiers(zero_frames=10, generator=self.generator)
        out = projector(tiers)
        self.assertEqual(out[0][:, :, :10].abs().max().item(), 0.0)

    def test_renorm_freezes_the_reference_then_restores_statistics(self):
        projector = HistoryProjector(mode="renorm")
        reference = history_tiers(generator=self.generator)
        first = projector(reference)
        for original, projected in zip(reference, first):
            torch.testing.assert_close(original, projected)
        self.assertEqual(projector.stats()["reference_call"], 1)

        drifted = history_tiers(generator=self.generator, scale=1.4, shift=0.8)
        out = projector(drifted)
        values = HistoryProjector._channel_values(
            out, [torch.ones(1, tier.shape[2], dtype=torch.bool) for tier in out]
        )
        ref_mean, ref_std = projector.reference
        torch.testing.assert_close(values.mean(dim=1), ref_mean, atol=2e-2, rtol=0.2)
        torch.testing.assert_close(values.std(dim=1), ref_std, atol=2e-2, rtol=0.2)

    def test_renorm_waits_for_a_fully_populated_window(self):
        projector = HistoryProjector(mode="renorm")
        projector(history_tiers(zero_frames=10, generator=self.generator))
        self.assertIsNone(projector.reference)
        projector(history_tiers(generator=self.generator))
        self.assertEqual(projector.stats()["reference_call"], 2)

    def test_near_constant_frame_is_still_projected(self):
        # A statistical padding test would treat a legitimate near-constant
        # frame as padding and silently skip it. Padding detection is
        # structural, so only exactly-zero frames are passed through.
        projector = HistoryProjector(mode="quantize", step=0.5)
        tiers = history_tiers(generator=self.generator)
        tiers[0][:, :, 0] = 0.01
        projector(tiers)
        self.assertEqual(projector.stats()["num_frames_padding"], 0)

    def test_effect_counters_expose_a_zero_effect_projection(self):
        # num_applied only proves the branch ran. A projection of already
        # projected history changes nothing and must be visible as such.
        step = 0.5
        first = HistoryProjector(mode="quantize", step=step)
        projected = first(history_tiers(generator=self.generator))
        self.assertGreater(first.stats()["num_changed"], 0)
        self.assertGreater(first.stats()["max_abs_delta"], 0.0)

        second = HistoryProjector(mode="quantize", step=step)
        second(projected)
        stats = second.stats()
        self.assertEqual(stats["num_applied"], 1)
        self.assertEqual(stats["num_changed"], 0)
        self.assertEqual(stats["max_abs_delta"], 0.0)

    def test_padding_frames_are_counted(self):
        projector = HistoryProjector(mode="quantize", step=0.5)
        projector(history_tiers(zero_frames=10, generator=self.generator))
        stats = projector.stats()
        self.assertEqual(stats["num_frames_seen"], 19)
        self.assertEqual(stats["num_frames_padding"], 10)

    def test_invalid_configurations_are_rejected(self):
        with self.assertRaises(ValueError):
            HistoryProjector(mode="quantize", step=0.5, padding_tol=-1.0)
        with self.assertRaises(ValueError):
            HistoryProjector(mode="bogus")
        with self.assertRaises(ValueError):
            HistoryProjector(mode="quantize", step=0.0)
        with self.assertRaises(ValueError):
            HistoryProjector(mode="codebook")
        with self.assertRaises(ValueError):
            HistoryProjector(mode="codebook", codebook=torch.randn(4))
        with self.assertRaises(ValueError):
            HistoryProjector(mode="quantize", step=0.5, alpha=1.5)

    def test_shapes_and_dtype_are_preserved(self):
        projector = HistoryProjector(mode="quantize", step=0.25)
        tiers = history_tiers(generator=self.generator)
        for original, projected in zip(tiers, projector(tiers)):
            self.assertEqual(original.shape, projected.shape)
            self.assertEqual(original.dtype, projected.dtype)


if __name__ == "__main__":
    unittest.main()
