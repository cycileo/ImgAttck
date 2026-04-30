from imgattck.preprocess import QwenImageSpec
from imgattck.prompting import manual_prompt_inputs, text_processor_inputs


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


def test_manual_prompt_can_disable_reasoning_prefix():
    class RecordingTokenizer(FakeTokenizer):
        text = ""

        def __call__(self, text, return_tensors, add_special_tokens):
            self.text = text
            return super().__call__(text, return_tensors, add_special_tokens)

    tokenizer = RecordingTokenizer()
    spec = QwenImageSpec(height=224, width=224, patch_size=16, temporal_patch_size=2, merge_size=2)

    manual_prompt_inputs(tokenizer, "Describe.", spec, enable_thinking=False)

    assert tokenizer.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_text_processor_inputs_uses_text_only_messages():
    class RecordingProcessor:
        messages = None
        kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            import torch

            self.messages = messages
            self.kwargs = kwargs
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

    processor = RecordingProcessor()

    batch = text_processor_inputs(processor, "Hello", enable_thinking=False)

    assert batch["input_ids"].tolist() == [[1, 2, 3]]
    assert processor.messages == [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    assert processor.kwargs["enable_thinking"] is False
    assert processor.kwargs["return_tensors"] == "pt"
