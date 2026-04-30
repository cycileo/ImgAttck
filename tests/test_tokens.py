from imgattck.tokens import check_target_strings, require_single_token_targets


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        pieces = {" yes": [1], " no": [2], "two tokens": [3, 4], "": []}
        return pieces[text]

    def decode(self, token_ids, skip_special_tokens=False):
        return f"decoded:{','.join(str(token_id) for token_id in token_ids)}"


def test_check_target_strings_reports_single_tokens():
    checks = check_target_strings(FakeTokenizer(), [" yes", "two tokens"])

    assert checks[0].is_single_token
    assert not checks[1].is_single_token
    assert checks[1].token_ids == [3, 4]


def test_require_single_token_targets_raises_for_multi_token():
    try:
        require_single_token_targets(FakeTokenizer(), [" yes", "two tokens"])
    except ValueError as exc:
        assert "two tokens" in str(exc)
    else:
        raise AssertionError("Expected multi-token target to fail")
