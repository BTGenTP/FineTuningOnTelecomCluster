"""
Fine-tune LLM with QLoRA on NAV4RAIL BT dataset.

Supported models: llama3_8b, mistral_7b

Usage:
    python finetune_llama3_nav4rail.py --model llama3_8b --dataset dataset.jsonl
    python finetune_llama3_nav4rail.py --model mistral_7b --dataset dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

# ─── NAV4RAIL system prompt (constant, embedded) ────────────────────────────

SYSTEM_PROMPT = """\
Tu es un expert en robotique ferroviaire NAV4RAIL. Genere un Behavior Tree XML BehaviorTree.CPP pour la mission decrite.

FORMAT :
- <root BTCPP_format="4" main_tree_to_execute="nom">
- Multi-<BehaviorTree ID="..."> interconnectes via <SubTreePlus __autoremap="true">
- <Action name="NOM" ID="Skill" port="{var}"/>  <Condition name="NOM" ID="Skill"/>
- Controle : Sequence, Fallback, ReactiveFallback, Repeat(num_cycles="-1")
- Chaque noeud a name="DESCRIPTION EN MAJUSCULES", ports blackboard {variable}

ARCHITECTURE :
principal -> Sequence(preparation + execution via SubTreePlus)
preparation -> LoadMission + MissionStructureValid + calculate_path + PassAdvancedPath + PassMission + GenerateMissionSequence
calculate_path -> Fallback(Repeat(-1)(UpdateCurrentGeneratedActivity/ProjectPointOnNetwork/CreatePath/AgregatePath), MissionFullyTreated)
execution -> ReactiveFallback(Repeat(-1)(Fallback motion_selector), MissionTerminated)

CHOIX DES MOTION SUBTREES (CRUCIAL — adapter a la mission) :
Transport (TOUJOURS inclure) :
  move(type=0 Move), deccelerate(type=1 Deccelerate), reach_and_stop(type=2 MoveAndStop+SignalAndWaitForOrder), pass(type=3 Move threshold=3), reach_stop_no_wait(type=4 MoveAndStop)
Inspection AVEC controle (si 'verifier'/'controler' les mesures) — AJOUTER :
  move_and_inspect(type=10): Pause + ManageMeasurements(start) + Move
  deccel_and_inspect(type=11): Deccelerate (mesures en cours)
  reach_stop_inspecting(type=12): MoveAndStop + ManageMeasurements(stop) + AnalyseMeasurements + Fallback(MeasurementsQualityValidated, PassDefectsLocalization) + GenerateCorrectiveSubSequence + InsertCorrectiveSubSequence
  pass_stop_inspecting(type=13): Move(pass) + ManageMeasurements(stop) + Fallback(AnalyseMeasurements, MeasurementsEnforcedValidated)
  reach_stop_inspect_no_wait(type=14): comme type=12 sans SignalAndWaitForOrder
Inspection SANS controle (mesures 'a la volee') — AJOUTER :
  types 10-14 avec ManageMeasurements MAIS SANS AnalyseMeasurements/MeasurementsQualityValidated

Condition dans Fallback : MeasurementsQualityValidated TOUJOURS enfant direct de Fallback.

VARIETE STRUCTURELLE (IMPORTANT) :
- Adapte le name= de chaque noeud a la mission specifique (element inspecte, km, contexte).
- Tu PEUX varier l'ordre des subtrees dans le MOTION SELECTOR.
- Tu PEUX ajouter des Pause(duration) entre certaines etapes quand c'est pertinent.
- Tu PEUX omettre certains subtrees optionnels (ex: pass type=3 ou reach_stop_no_wait type=4 ne sont pas toujours necessaires).
- Les durations de Pause peuvent varier (1.0 a 5.0).
- Les messages de SignalAndWaitForOrder doivent refleter la mission.
- Ajoute des commentaires XML <!-- ... --> decrivant la mission.
Reponds uniquement avec le XML.

Skills (28, 5 familles) :

PREPARATION :
- LoadMission (mission_file_path)
- MissionStructureValid [Condition]
- UpdateCurrentGeneratedActivity (type, origin_sph, target_sph, forbidden_atoms_out)
- ProjectPointOnNetwork (point_in, point_out)
- CreatePath (origin, target, forbidden_atoms, path)
- AgregatePath (path)
- MissionFullyTreated [Condition] (type)
- PassAdvancedPath (adv_path)
- PassMission (mission)
- GenerateMissionSequence (mission, mission_sequence)
- GenerateCorrectiveSubSequence (defects)
- InsertCorrectiveSubSequence

MOTION :
- MissionTerminated [Condition]
- CheckCurrentStepType [Condition] (type_to_be_checked: 0=move 1=decel 2=reach_stop 3=pass 4=no_wait 10-14=inspection)
- PassMotionParameters (motion_params)
- Move (threshold_type: 1=normal 3=pass, motion_params)
- UpdateCurrentExecutedStep
- Deccelerate (motion_params)
- MoveAndStop (motion_params)
- SignalAndWaitForOrder (message)
- IsRobotPoseProjectionActive [Condition] (adv_path, pub_proj)
- Pause (duration)

INSPECTION :
- ManageMeasurements
- AnalyseMeasurements
- MeasurementsQualityValidated [Condition]
- PassDefectsLocalization (defects)
- MeasurementsEnforcedValidated [Condition]

SIMULATION :
- SimulationStarted [Condition]"""

# ─── Model configs ───────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "llama3_8b": {
        "hf_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "response_anchor": "assistant<|end_header_id|>",
        "chat_template": True,  # use tokenizer.apply_chat_template
        "bf16": True,
        "default_epochs": 10,
        "default_batch": 1,
        "default_grad_accum": 16,
    },
    "mistral_7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "response_anchor": "[/INST]",
        "chat_template": False,  # manual [INST]...[/INST] formatting
        "bf16": False,  # Mistral 7B v0.2 uses fp16
        "default_epochs": 10,
        "default_batch": 2,
        "default_grad_accum": 8,
    },
}

LORA_R = 16
LORA_ALPHA = 32
MAX_SEQ_LEN = 8192
LR = 2e-4


def _format_mistral(mission: str, xml: str) -> str:
    """Format as Mistral [INST]...[/INST] prompt."""
    return f"<s>[INST] {SYSTEM_PROMPT}\n\nMission : {mission} [/INST]\n{xml} </s>"


def load_dataset(path: Path, tokenizer, model_cfg: dict) -> Dataset:
    """Load JSONL dataset and format for the target model."""
    use_chat_template = model_cfg["chat_template"]
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mission = obj.get("mission", "").strip()
            xml = obj.get("xml", "").strip()
            if not mission or not xml:
                continue

            if use_chat_template:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Mission : {mission}"},
                    {"role": "assistant", "content": xml},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                text = _format_mistral(mission, xml)
            rows.append({"text": text})

    print(f"Loaded {len(rows)} samples from {path}")
    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # Token length stats
    sample_lens = []
    for r in rows[:50]:
        toks = tokenizer(r["text"], truncation=False)["input_ids"]
        sample_lens.append(len(toks))
    if sample_lens:
        print(
            f"  Token lengths (first 50): min={min(sample_lens)}, avg={sum(sample_lens) // len(sample_lens)}, max={max(sample_lens)}"
        )

    return split


def load_model_and_tokenizer(model_cfg: dict):
    """Load model with 4-bit quantization + LoRA."""
    hf_id = model_cfg["hf_id"]
    use_bf16 = model_cfg["bf16"]
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Loading {hf_id} in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # Verify LoRA targets exist
    present = {name.split(".")[-1] for name, _ in model.named_modules() if name}
    targets = [t for t in model_cfg["lora_targets"] if t in present]
    if not targets:
        raise RuntimeError(f"No LoRA targets found. Present: {sorted(present)[:20]}")
    print(f"  LoRA targets: {targets}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune on NAV4RAIL BT dataset"
    )
    p.add_argument("--model", choices=sorted(MODEL_CONFIGS.keys()), required=True)
    p.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset")
    p.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for adapter"
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_cfg = MODEL_CONFIGS[args.model]

    epochs = args.epochs or model_cfg["default_epochs"]
    batch_size = args.batch_size or model_cfg["default_batch"]
    grad_accum = args.grad_accum or model_cfg["default_grad_accum"]
    use_bf16 = model_cfg["bf16"]

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_path.parent / "outputs" / f"nav4rail_{args.model}_lora"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_cfg)
    split = load_dataset(dataset_path, tokenizer, model_cfg)

    # Completion-only collator: mask loss on everything before the response
    collator = DataCollatorForCompletionOnlyLM(
        response_template=model_cfg["response_anchor"],
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=args.lr,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    print(f"\n{'=' * 60}")
    print(f"Model: {model_cfg['hf_id']}")
    print(
        f"Training: {epochs} epochs, batch={batch_size}×{grad_accum}={batch_size * grad_accum}"
    )
    print(f"  LR: {args.lr}, max_seq_len: {args.max_seq_len}")
    print(f"  LoRA: r={args.lora_r}, alpha={LORA_ALPHA}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}\n")

    trainer.train()

    # Save adapter
    adapter_dir = out_dir / "lora_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nAdapter saved: {adapter_dir}")


if __name__ == "__main__":
    main()
