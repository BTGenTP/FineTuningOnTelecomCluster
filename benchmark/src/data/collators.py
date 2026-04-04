from __future__ import annotations

from typing import Any


def build_completion_only_collator(tokenizer: Any, response_template: str = "XML:\n") -> Any:
    from trl import DataCollatorForCompletionOnlyLM

    return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
