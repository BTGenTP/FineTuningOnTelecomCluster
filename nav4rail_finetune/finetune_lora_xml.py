"""
Fine-tuning QLoRA pour NAV4RAIL — Mission NL → BehaviorTree XML
=================================================================
Modèle  : Mistral-7B-Instruct-v0.2 (ou TinyLlama-1.1B-Chat pour baseline)
Méthode : QLoRA (4-bit quantization + LoRA adapters)
GPU     : Tesla P100-PCIE-16GB (cluster Telecom Paris)
Dataset : dataset_nav4rail_500.jsonl — 500 paires (mission, XML)

Usage :
    python finetune_lora_xml.py --model mistral              # Mistral-7B (recommandé)
    python finetune_lora_xml.py --model tinyllama            # Baseline rapide
    python finetune_lora_xml.py --model mistral --eval-only  # Inférence seule
    python finetune_lora_xml.py --model mistral --eval-only --constrained
                                                             # + décodage contraint GBNF
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from validate_bt import validate_bt   # Validateur multi-niveaux (L1 + L2 + L3)

# ─── Configuration ───────────────────────────────────────────────────────────

MODELS = {
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_len": 1024,
        "batch_size": 2,
        "grad_accum": 8,
        "epochs": 5,
        "lr": 2e-4,
    },
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_r": 8,
        "lora_alpha": 16,
        "max_seq_len": 1024,
        "batch_size": 4,
        "grad_accum": 4,
        "epochs": 8,
        "lr": 3e-4,
    },
}

DATASET_PATH = Path(__file__).parent / "dataset_nav4rail_500.jsonl"
OUTPUT_DIR   = Path(__file__).parent / "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_info():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU {used:.1f}/{total:.1f} GB"
    return "CPU only"


# ─── Chargement du dataset ───────────────────────────────────────────────────

def load_dataset(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    # Le champ "prompt" contient déjà le format Mistral instruction complet :
    # <s>[INST] {system + skills + mission} [/INST] {XML} </s>
    dataset = Dataset.from_list([{"text": r["prompt"]} for r in records])

    # Split 90/10 train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    log(f"Dataset : {len(split['train'])} train / {len(split['test'])} eval")
    return split


# ─── Chargement du modèle (QLoRA) ───────────────────────────────────────────

def load_model_and_tokenizer(model_key: str):
    cfg = MODELS[model_key]
    hf_id = cfg["hf_id"]

    log(f"Chargement tokenizer : {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # QLoRA : quantification 4-bit avec bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 (meilleur pour LLM)
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,     # Double quantification → économie mémoire
    )

    log(f"Chargement modèle 4-bit : {hf_id} [{mem_info()}]")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    log(f"Modèle chargé [{mem_info()}]")

    # Préparation pour le k-bit training (cast des couches normalization)
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_targets"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log(f"Paramètres LoRA : {trainable:,} entraînables / {total:,} total "
        f"({100*trainable/total:.2f}%)")

    return model, tokenizer, cfg


# ─── Entraînement ────────────────────────────────────────────────────────────

def train(model_key: str):
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer, cfg = load_model_and_tokenizer(model_key)
    split = load_dataset(DATASET_PATH)

    output_path = OUTPUT_DIR / f"nav4rail_{model_key}_lora"
    output_path.mkdir(parents=True, exist_ok=True)

    # Réponse-only loss : on entraîne seulement sur la partie [/INST]...
    # Le collateur ignore les tokens de l'instruction.
    response_template = "[/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,                      # fp16 sur P100 (pas de bf16)
        bf16=False,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",               # pas de wandb sur le cluster
        optim="paged_adamw_8bit",       # optimizer 8-bit pour économiser la mémoire
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_len"],
        packing=False,
    )

    log(f"Début entraînement [{mem_info()}]")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log(f"Entraînement terminé en {elapsed/60:.1f} min [{mem_info()}]")

    # Sauvegarde des adapters LoRA
    adapter_path = output_path / "lora_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log(f"Adapter LoRA sauvegardé : {adapter_path}")

    return trainer.model, tokenizer, cfg


# ─── Inférence ───────────────────────────────────────────────────────────────

SKILLS_DOC = """Skills disponibles :
- GetMission        : Récupère et valide les paramètres de la mission
- CalculatePath     : Calcule le chemin optimal vers la destination
- Move              : Déplacement du robot le long de la voie ferrée
- Decelerate        : Décélération progressive et contrôlée
- ManageMeasurement : Effectue des mesures (géométrie, alignement, thermique...)
- CheckObstacle     : Vérifie l'absence d'obstacles sur la voie (retourne SUCCESS si libre)
- Alert             : Envoie une alerte ou un rapport au système central
- Stop              : Arrêt complet et sécurisé du robot"""

SYSTEM_PROMPT = (
    "Tu es un expert en robotique ferroviaire NAV4RAIL. "
    "Génère un Behavior Tree au format XML BehaviorTree.CPP v4 "
    "correspondant exactement à la mission décrite. "
    "Utilise uniquement les skills du catalogue fourni. "
    "Réponds uniquement avec le XML, sans explication."
)


def _build_prompt(mission: str) -> str:
    instruction = f"{SYSTEM_PROMPT}\n\n{SKILLS_DOC}\n\nMission : {mission}"
    return f"<s>[INST] {instruction} [/INST]"


def _extract_xml(decoded: str) -> str:
    """Extrait le bloc <root>...</root> de la sortie brute du modèle."""
    if "[/INST]" in decoded:
        raw = decoded.split("[/INST]", 1)[1].strip()
    else:
        raw = decoded.strip()
    match = re.search(r"(<root\b.*?</root>)", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw


def generate_xml(model, tokenizer, mission: str, max_new_tokens: int = 600) -> str:
    """Génération XML libre (décodage glouton standard)."""
    prompt = _build_prompt(mission)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return _extract_xml(tokenizer.decode(out[0], skip_special_tokens=True))


def generate_xml_constrained(model, tokenizer, mission: str,
                              max_new_tokens: int = 600) -> str:
    """
    Génération XML avec décodage contraint (grammaire GBNF via lm-format-enforcer).

    À chaque étape, seuls les tokens compatibles avec la grammaire NAV4RAIL
    sont autorisés → zéro hallucination de nom de skill garantie.

    Requiert : pip install lm-format-enforcer
    """
    from nav4rail_grammar import get_prefix_fn

    prefix_fn = get_prefix_fn(tokenizer)
    prompt = _build_prompt(mission)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_fn,   # ← contrainte grammaticale
        )
    return _extract_xml(tokenizer.decode(out[0], skip_special_tokens=True))


# ─── Évaluation ──────────────────────────────────────────────────────────────

TEST_MISSIONS = [
    "Inspecte la section de voie au km 30",
    "Mesure la géométrie de la voie sur 3 km depuis le km 12",
    "Navigue en mode sécurisé vers le secteur nord",
    "Effectue une patrouille entre km 0 et km 5 avec rapport",
    "Va au dépôt principal après l'inspection",
    "Certifie la section B après les travaux de maintenance",
    "Contrôle complet avec alerte si défaut détecté au km 25",
    "Mesure les paramètres thermiques entre km 8 et km 10",
    "Inspecte le tunnel au km 33 avec vérification obstacle",
    "Déplace-toi vers le point de chargement et attends",
]


def evaluate(model, tokenizer, n_samples: int = 10, constrained: bool = False):
    """
    Évalue le modèle sur des missions hors dataset.

    Utilise validate_bt() (3 niveaux : syntaxique, structurel, sémantique).
    Affiche pour chaque mission : statut, score, avertissements et XML.
    """
    mode = "contraint (GBNF)" if constrained else "libre"
    print("\n" + "─" * 70)
    print(f"ÉVALUATION — Génération XML [{mode}] sur missions hors dataset")
    print("─" * 70)

    valid_count   = 0
    warning_count = 0
    scores        = []
    all_errors    = []

    for mission in TEST_MISSIONS[:n_samples]:
        if constrained:
            xml = generate_xml_constrained(model, tokenizer, mission)
        else:
            xml = generate_xml(model, tokenizer, mission)

        vr = validate_bt(xml)

        status   = "✓" if vr.valid else "✗"
        warn_tag = f"  [{len(vr.warnings)}W]" if vr.warnings else ""
        score_tag = f"  score={vr.score:.1f}"

        print(f"\n[{status}] {mission}{warn_tag}{score_tag}")

        if vr.valid:
            valid_count += 1
            scores.append(vr.score)
            if vr.warnings:
                warning_count += 1
                for w in vr.warnings:
                    print(f"    {w}")
            print(xml[:300] + ("..." if len(xml) > 300 else ""))
        else:
            all_errors.extend(vr.errors)
            for e in vr.errors:
                print(f"    ERREUR : {e}")
            print(f"    Généré : {xml[:200]}")

    total     = n_samples
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "─" * 70)
    print(f"Résultat        : {valid_count}/{total} BTs valides "
          f"({100 * valid_count / total:.0f}%)")
    print(f"Score moyen     : {avg_score:.2f} / 1.0")
    print(f"Avec warnings   : {warning_count} / {valid_count} valides")
    print("─" * 70)

    return {
        "valid":    valid_count,
        "invalid":  total - valid_count,
        "warnings": warning_count,
        "avg_score": avg_score,
        "errors":   all_errors,
    }


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning QLoRA NAV4RAIL")
    parser.add_argument("--model", choices=["mistral", "tinyllama"],
                        default="mistral", help="Modèle à fine-tuner")
    parser.add_argument("--eval-only", action="store_true",
                        help="Inférence uniquement (charge l'adapter sauvegardé)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Chemin vers l'adapter LoRA pour --eval-only")
    parser.add_argument("--constrained", action="store_true",
                        help="Décodage contraint par grammaire GBNF (nécessite "
                             "lm-format-enforcer)")
    args = parser.parse_args()

    log(f"=== NAV4RAIL Fine-Tuning QLoRA ===")
    log(f"Modèle : {args.model} | Device : {DEVICE} | {mem_info()}")

    if args.eval_only:
        # Chargement de l'adapter seul pour inférence
        from peft import PeftModel
        adapter_path = args.adapter_path or str(
            OUTPUT_DIR / f"nav4rail_{args.model}_lora" / "lora_adapter"
        )
        cfg = MODELS[args.model]
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["hf_id"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        log("Adapter LoRA chargé.")
    else:
        model, tokenizer, _ = train(args.model)

    evaluate(model, tokenizer, constrained=args.constrained)
    log("Terminé.")


if __name__ == "__main__":
    main()
