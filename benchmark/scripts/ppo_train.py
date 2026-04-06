from __future__ import annotations

import argparse
import random
from pathlib import Path

from _paths import ensure_sys_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train with PPO (TRL) using deterministic validator reward.")
    p.add_argument("--config", type=str, default="configs/ppo.yaml")
    p.add_argument("--prompts", type=str, default=None, help="Optional JSONL with {'mission': ...} rows.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", type=str, default=None, help="Override config.output_root.")
    p.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional PEFT adapter path (e.g. SFT LoRA) to initialize policy and frozen ref.",
    )
    return p.parse_args()


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        from _jsonl import read_jsonl

        rows = list(read_jsonl(args.prompts))
        missions = [str(r.get("mission", "")).strip() for r in rows]
        return [m for m in missions if m]
    from src.data.synthetic_generator import PROMPT_TEMPLATES

    random.seed(args.seed)
    return [random.choice(PROMPT_TEMPLATES) for _ in range(args.n)]


def main() -> int:
    ensure_sys_path()
    args = parse_args()

    import torch

    from src.config_loader import load_experiment_config
    from src.data.catalog import default_catalog_path, load_catalog
    from src.data.formatting import render_system_prompt
    from src.evaluation.run_manifest import write_manifest_for_run
    from src.evaluation.runner import create_run_paths
    from src.models.factory import load_model_bundle
    from src.rewards.reward_fn import reward_from_xml
    from src.xml_utils import extract_root_xml

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))
    if args.adapter:
        cfg.peft.adapter_path = args.adapter

    paths = create_run_paths(cfg.output_root)
    paths.experiment_json.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    write_manifest_for_run(
        paths,
        config_path=Path(args.config).resolve(),
        cfg=cfg,
        extra={"adapter_cli": args.adapter, "prompts_file": args.prompts, "script": "ppo_train.py"},
    )

    catalog = load_catalog(cfg.catalog_path or default_catalog_path())
    system_prompt = render_system_prompt(catalog, include_schema=cfg.prompt.include_schema)

    def build_prompt(mission: str) -> str:
        return f"{system_prompt}\n\nMission:\n{mission.strip()}\n\nXML:\n"

    try:
        # TRL moved this wrapper between modules across versions.
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead  # type: ignore
    except Exception:
        try:
            from trl import AutoModelForCausalLMWithValueHead  # type: ignore
        except Exception as exc:
            raise SystemExit(f"TRL ValueHead wrapper not available: {exc}")

    if not cfg.peft.method and not cfg.peft.adapter_path:
        raise SystemExit("PPO LoRA: set peft.method (e.g. lora) and/or peft.adapter_path / --adapter for SFT init.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_inner, tokenizer = load_model_bundle(cfg.model, cfg.peft)
    model_inner.train()
    try:
        model_inner.config.use_cache = False
    except Exception:
        pass
    if hasattr(model_inner, "gradient_checkpointing_enable"):
        try:
            model_inner.gradient_checkpointing_enable()
        except Exception:
            pass

    from peft import PeftModel

    if not isinstance(model_inner, PeftModel):
        raise SystemExit(
            "PPO on a single GPU expects a PEFT model: TRL uses ref_model=None and KL vs the frozen base "
            "via disable_adapter(). Without PEFT, a full duplicate reference model would not fit typical 16 GiB cards."
        )

    if device.type == "cuda":
        model_inner = model_inner.to(device)
        torch.cuda.empty_cache()

    model = AutoModelForCausalLMWithValueHead(model_inner).to(device)
    # TRL's wrapper sets `is_peft_model` only when constructed via `.from_pretrained(...)`.
    # We wrap an already-loaded (PEFT) model, so ensure the attribute exists for TRL internals.
    if not hasattr(model, "is_peft_model"):
        model.is_peft_model = isinstance(model_inner, PeftModel)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    def _gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=-1)
        return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    def _policy_logprobs_and_values(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lm_logits, _loss, values = model(input_ids=input_ids, attention_mask=attention_mask)
        token_logprobs = _gather_logprobs(lm_logits[:, :-1, :], input_ids[:, 1:])
        return token_logprobs, values

    def _ref_logprobs(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inner = getattr(model, "pretrained_model", None) or model_inner
        if hasattr(inner, "disable_adapter"):
            with inner.disable_adapter():
                out = inner(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                ref_logits = out.logits
        else:
            out = inner(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            ref_logits = out.logits
        if ref_logits.dtype != torch.float32:
            ref_logits = ref_logits.float()
        return _gather_logprobs(ref_logits[:, :-1, :], input_ids[:, 1:])

    # Minimal PPO hyperparams (stable defaults; config file does not specify them yet).
    cliprange = float(getattr(cfg.training, "cliprange", 0.2) or 0.2)
    vf_coef = float(getattr(cfg.training, "vf_coef", 0.5) or 0.5)
    kl_coef = float(getattr(cfg.training, "kl_coef", 0.05) or 0.05)
    gamma = float(getattr(cfg.training, "gamma", 1.0) or 1.0)
    lam = float(getattr(cfg.training, "lam", 0.95) or 0.95)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.training.learning_rate),
    )

    missions = _load_prompts(args)
    random.seed(args.seed)

    rows = []
    max_seq_length = int(getattr(cfg.training, "max_seq_length", 2048) or 2048)
    desired_new = int(getattr(cfg.generation, "max_new_tokens", 256) or 256)

    for epoch in range(args.epochs):
        random.shuffle(missions)
        for mission in missions:
            query = build_prompt(mission)
            prompt_max_len = max(32, max_seq_length - max(1, desired_new) - 1)
            enc = tokenizer(query, return_tensors="pt", truncation=True, max_length=prompt_max_len)
            query_ids = enc.input_ids.to(device)
            attn = getattr(enc, "attention_mask", None)
            attn = attn.to(device) if attn is not None else torch.ones_like(query_ids)
            gen_kw: dict = {
                "attention_mask": attn,
                "max_new_tokens": max(1, min(desired_new, max_seq_length - int(query_ids.shape[-1]) - 1)),
                "do_sample": cfg.generation.do_sample,
                "temperature": cfg.generation.temperature,
                "top_p": cfg.generation.top_p,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if int(getattr(cfg.generation, "top_k", -1)) > 0:
                gen_kw["top_k"] = int(cfg.generation.top_k)
            with torch.no_grad():
                seq = model.generate(input_ids=query_ids, **gen_kw)
            seq = seq.to(device)
            response_ids = seq[0]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            xml = extract_root_xml(response_text) or response_text.strip()

            score, breakdown, _report = reward_from_xml(
                xml_text=xml,
                catalog_path=cfg.catalog_path,
                xsd_path=cfg.xsd_path,
                strict=True,
                constraints_dir=str(Path("constraints").resolve()),
            )

            # --- PPO update (single-sample, on-policy) ---
            context_len = int(query_ids.shape[1])
            if int(response_ids.shape[0]) <= context_len:
                continue
            # `generate()` doesn't pad; attention mask is all ones.
            qr = response_ids.unsqueeze(0)
            qr_attn = torch.ones_like(qr, device=device)

            token_logprobs, values = _policy_logprobs_and_values(qr, qr_attn)

            # Align to response tokens: token logprob i corresponds to token at position i+1.
            start = max(0, context_len - 1)
            end = int(qr.shape[1] - 1)
            old_logprobs = token_logprobs[:, start:end].detach()
            logprobs = token_logprobs[:, start:end]
            vpred = values[:, start:end]
            # Computing reference logprobs by temporarily disabling LoRA adapters can interact badly
            # with Transformer gradient checkpointing (shape-mismatch CheckpointError during recompute).
            # For stability on single-GPU runs, use the policy itself as the reference (KL=0).
            ref_lp = old_logprobs
            resp_len = int(logprobs.shape[1])
            if resp_len <= 0:
                continue

            kl = old_logprobs - ref_lp
            rewards = -kl_coef * kl
            rewards[:, -1] = rewards[:, -1] + float(score)

            with torch.no_grad():
                v_det = vpred.detach()
                adv = torch.zeros_like(rewards)
                lastgaelam = torch.zeros((rewards.shape[0],), device=device)
                for t in range(resp_len - 1, -1, -1):
                    next_v = v_det[:, t + 1] if t + 1 < resp_len else 0.0
                    delta = rewards[:, t] + gamma * next_v - v_det[:, t]
                    lastgaelam = delta + gamma * lam * lastgaelam
                    adv[:, t] = lastgaelam
                returns = adv + v_det
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = torch.exp(logprobs - old_logprobs)
            pg_losses1 = -adv * ratio
            pg_losses2 = -adv * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
            policy_loss = torch.max(pg_losses1, pg_losses2).mean()
            value_loss = 0.5 * (vpred - returns).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            stats = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "approx_kl": float((old_logprobs - logprobs).mean().detach().cpu()),
                "clipfrac": float((torch.abs(ratio - 1.0) > cliprange).float().mean().detach().cpu()),
            }
            rows.append(
                {
                    "epoch": epoch,
                    "mission_preview": mission[:80],
                    "reward": float(score),
                    "pass_l1": breakdown.pass_l1,
                    "pass_l2": breakdown.pass_l2,
                    "pass_l3": breakdown.pass_l3,
                    "ppo_stats": stats,
                }
            )

    out_dir = Path(cfg.training.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inner = getattr(model, "pretrained_model", None)
    if inner is not None:
        inner.save_pretrained(out_dir)
    else:
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    from src.evaluation.metrics import render_markdown_table
    import json

    paths.metrics_json.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths.summary_md.write_text(render_markdown_table(rows[-50:]) + "\n", encoding="utf-8")
    print("run_dir:", paths.run_dir)
    print("saved_model:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
