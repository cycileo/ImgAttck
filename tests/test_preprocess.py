import torch

from imgattck.preprocess import QwenImageSpec, differentiable_qwen_preprocess


def test_differentiable_preprocess_shapes_and_grid():
    spec = QwenImageSpec(height=224, width=224, patch_size=16, temporal_patch_size=2, merge_size=2)
    image = torch.full((1, 3, 224, 224), 0.5)

    processed = differentiable_qwen_preprocess(image, spec)

    assert processed.pixel_values.shape == (196, 1536)
    assert processed.image_grid_thw.tolist() == [[1, 14, 14]]


def test_differentiable_preprocess_matches_qwen_processor_packaging():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    torch.manual_seed(0)
    spec = QwenImageSpec(height=224, width=224, patch_size=16, temporal_patch_size=2, merge_size=2)
    image = torch.rand(3, 224, 224)
    ours = differentiable_qwen_preprocess(image, spec)

    processor = Qwen2VLImageProcessor(
        size={"shortest_edge": 224 * 224, "longest_edge": 224 * 224},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
    )
    native = processor.preprocess([image * 255.0], return_tensors="pt")

    torch.testing.assert_close(ours.pixel_values.cpu(), native["pixel_values"], atol=1e-5, rtol=1e-5)
    assert ours.image_grid_thw.cpu().tolist() == native["image_grid_thw"].tolist()
