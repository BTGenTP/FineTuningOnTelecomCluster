from __future__ import annotations

from typing import Any


def build_completion_only_collator(tokenizer: Any, response_template: str = "XML:\n") -> Any:
    """TRL < 1.0 only — `DataCollatorForCompletionOnlyLM` was removed in TRL 1.0."""
    from trl import DataCollatorForCompletionOnlyLM

    return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
