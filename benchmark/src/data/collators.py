from __future__ import annotations

from typing import Any


def build_completion_only_collator(tokenizer: Any, response_template: str = "XML:\n") -> Any:
    """Completion-only loss collator for TRL versions that provide it.

    Note: This collator is not consistently exported from `trl.__init__` across TRL releases.
    """
    try:
        from trl import DataCollatorForCompletionOnlyLM  # type: ignore
    except Exception:
        try:
            from trl.trainer.utils import DataCollatorForCompletionOnlyLM  # type: ignore
        except Exception:
            try:
                from trl.trainer.sft_trainer import DataCollatorForCompletionOnlyLM  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "DataCollatorForCompletionOnlyLM not found in TRL. "
                    "If you're on TRL>=1.0, use SFTConfig(completion_only_loss=True) path. "
                    "Otherwise pin `trl>=0.26.0,<0.29.0` as in requirements.txt."
                ) from exc

    return DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
