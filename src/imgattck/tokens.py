from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenCheck:
    text: str
    token_ids: list[int]
    decoded: str

    @property
    def is_single_token(self) -> bool:
        return len(self.token_ids) == 1


def check_target_strings(tokenizer: object, target_strings: list[str]) -> list[TokenCheck]:
    checks: list[TokenCheck] = []
    for text in target_strings:
        token_ids = _encode(tokenizer, text)
        decoded = _decode(tokenizer, token_ids)
        checks.append(TokenCheck(text=text, token_ids=token_ids, decoded=decoded))
    return checks


def require_single_token_targets(tokenizer: object, target_strings: list[str]) -> list[int]:
    if not target_strings:
        raise ValueError("target_strings must contain at least one target.")
    checks = check_target_strings(tokenizer, target_strings)
    failures = [check for check in checks if not check.is_single_token]
    if failures:
        details = "; ".join(f"{item.text!r} -> {item.token_ids}" for item in failures)
        raise ValueError(f"All target strings must encode to exactly one token: {details}")
    return [check.token_ids[0] for check in checks]


def token_report(checks: list[TokenCheck]) -> list[dict[str, object]]:
    return [
        {
            "text": check.text,
            "token_ids": check.token_ids,
            "decoded": check.decoded,
            "is_single_token": check.is_single_token,
        }
        for check in checks
    ]


def _encode(tokenizer: object, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]
    result = tokenizer(text, add_special_tokens=False)  # type: ignore[operator]
    return list(result["input_ids"])


def _decode(tokenizer: object, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))  # type: ignore[attr-defined]
    return str(token_ids)
