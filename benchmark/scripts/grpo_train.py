from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

from _paths import ensure_sys_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train with GRPO (group relative policy optimization) using deterministic validator reward.")
    p.add_argument("--config", type=str, default="configs/grpo.yaml")
    p.add_argument("--prompts", type=str, default=None, help="Optional JSONL with {'mission': ...} rows.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--output-root", type=str, default=None, help="Override config.output_root.")
    p.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional PEFT adapter path (e.g. SFT LoRA) to initialize trainable policy and frozen ref.",
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
    from src.evaluation.runner import create_run_paths
    from src.methods.rl.grpo import _normalize_group
    from src.models.factory import load_model_bundle
    from src.rewards.reward_fn import reward_from_xml
    from src.xml_utils import extract_root_xml

    cfg = load_experiment_config(Path(args.config))
    if args.output_root:
        cfg.output_root = str(Path(args.output_root))
    if args.adapter:
        cfg.peft.adapter_path = args.adapter

    # Run dir & basic bookkeeping
    paths = create_run_paths(cfg.output_root)
    paths.experiment_json.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    from src.evaluation.run_manifest import write_manifest_for_run

    write_manifest_for_run(
        paths,
        config_path=Path(args.config).resolve(),
        cfg=cfg,
        extra={"adapter_cli": args.adapter, "prompts_file": args.prompts, "script": "grpo_train.py"},
    )

    catalog = load_catalog(cfg.catalog_path or default_catalog_path())
    system_prompt = render_system_prompt(catalog, include_schema=cfg.prompt.include_schema)

    def build_prompt(mission: str) -> str:
        return f"{system_prompt}\n\nMission:\n{mission.strip()}\n\nXML:\n"

    model, tokenizer = load_model_bundle(cfg.model, cfg.peft)
    model.train()

    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Clone on CPU (frees GPU) so deepcopy does not OOM; keep ref on CPU so two 7B fp16 weights are not both on a 16 GiB GPU.
    if train_device.type == "cuda":
        model_cpu = model.cpu()
        torch.cuda.empty_cache()
        ref_model = copy.deepcopy(model_cpu)
        model = model_cpu.to(train_device)
        del model_cpu
        torch.cuda.empty_cache()
    else:
        ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    device = train_device
    ref_on_cpu = train_device.type == "cuda"
    if not ref_on_cpu:
        ref_model = ref_model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    prompts = _load_prompts(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows: list[dict[str, Any]] = []
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(prompts)
        for mission in prompts:
            prompt_text = build_prompt(mission)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            prompt_len = prompt_ids.shape[-1]

            gen_token_ids: list[torch.Tensor] = []
            responses: list[str] = []
            rewards: list[float] = []

            for gi in range(args.group_size):
                with torch.no_grad():
                    gen_kwargs: dict[str, Any] = {
                        "input_ids": prompt_ids,
                        "max_new_tokens": cfg.generation.max_new_tokens,
                        "do_sample": cfg.generation.do_sample,
                        "temperature": cfg.generation.temperature,
                        "top_p": cfg.generation.top_p,
                        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                    }
                    if getattr(cfg.generation, "top_k", -1) is not None and int(cfg.generation.top_k) > 0:
                        gen_kwargs["top_k"] = int(cfg.generation.top_k)
                    gen_out = model.generate(**gen_kwargs)
                full = gen_out[0]
                gen_ids = full[prompt_len:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                xml = extract_root_xml(text) or text.strip()
                score, breakdown, _report = reward_from_xml(
                    xml_text=xml,
                    catalog_path=cfg.catalog_path,
                    xsd_path=cfg.xsd_path,
                    strict=True,
                    constraints_dir=str(Path("constraints").resolve()),
                )
                gen_token_ids.append(gen_ids)
                responses.append(xml)
                rewards.append(score)

            advantages = _normalize_group(rewards)

            # Policy gradient update on the sampled responses.
            # Approx KL(policy||ref) ~= E_policy[logp_policy - logp_ref] over sampled tokens.
            losses: list[torch.Tensor] = []
            for gen_ids, adv, score in zip(gen_token_ids, advantages, rewards):
                if gen_ids.numel() == 0:
                    continue
                full_ids = torch.cat([prompt_ids[0], gen_ids], dim=0).unsqueeze(0)
                with torch.set_grad_enabled(True):
                    out = model(full_ids)
                    if ref_on_cpu:
                        with torch.no_grad():
                            ref_out = ref_model(full_ids.cpu())
                        ref_logits = ref_out.logits[:, :-1, :].to(device)
                    else:
                        ref_out = ref_model(full_ids)
                        ref_logits = ref_out.logits[:, :-1, :]
                    logits = out.logits[:, :-1, :]
                    target = full_ids[:, 1:]

                    # Mask prompt tokens.
                    mask = torch.zeros_like(target, dtype=torch.bool)
                    mask[:, prompt_len - 1 :] = True  # response tokens start at prompt_len

                    logp = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    ref_logp = torch.log_softmax(ref_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)

                    logp_resp = logp[mask]
                    ref_logp_resp = ref_logp[mask]
                    if logp_resp.numel() == 0:
                        continue

                    mean_logp = logp_resp.mean()
                    mean_kl = (logp_resp - ref_logp_resp).mean()

                    loss = -(float(adv) * mean_logp) + (args.kl_coef * mean_kl)
                    losses.append(loss)

            if losses:
                loss_total = torch.stack(losses).mean()
                loss_total.backward()
                if (step + 1) % max(1, cfg.training.gradient_accumulation_steps) == 0:
                    optim.step()
                    optim.zero_grad(set_to_none=True)

                rows.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "mission_preview": mission[:80],
                        "group_size": args.group_size,
                        "reward_mean": float(sum(rewards) / max(1, len(rewards))),
                        "reward_max": float(max(rewards) if rewards else 0.0),
                        "reward_min": float(min(rewards) if rewards else 0.0),
                        "loss": float(loss_total.detach().cpu().item()),
                    }
                )
            step += 1

    # Save model/adapters.
    out_dir = Path(cfg.training.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    except Exception:
        pass

    # Persist summary.
    from src.evaluation.metrics import render_markdown_table

    paths.metrics_json.write_text(__import__("json").dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths.summary_md.write_text(render_markdown_table(rows[-50:]) + "\n", encoding="utf-8")
    print("run_dir:", paths.run_dir)
    print("saved_model:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

