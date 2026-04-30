from imgattck.preprocess import QwenImageSpec
from imgattck.prompting import manual_prompt_inputs


class FakeTokenizer:
    image_token_id = 99

    def __call__(self, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        import torch

        count = text.count("<|image_pad|>")
        return {
            "input_ids": torch.tensor([[1] + [99] * count + [2]]),
            "attention_mask": torch.ones((1, count + 2), dtype=torch.long),
        }


def test_manual_prompt_inputs_marks_image_tokens():
    spec = QwenImageSpec(height=224, width=224, patch_size=16, temporal_patch_size=2, merge_size=2)

    inputs = manual_prompt_inputs(FakeTokenizer(), "Describe.", spec)

    assert inputs["image_grid_thw"].tolist() == [[1, 14, 14]]
    assert int(inputs["mm_token_type_ids"].sum()) == spec.num_image_tokens
