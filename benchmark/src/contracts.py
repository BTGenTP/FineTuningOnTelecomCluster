from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


Severity = Literal["error", "warning", "info"]
ValidationLayer = Literal["L1", "L2", "L3"]


@dataclass(slots=True)
class ValidationIssue:
    layer: ValidationLayer
    severity: Severity
    code: str
    message: str
    xpath: Optional[str] = None
    node_tag: Optional[str] = None
    node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationSummary:
    issues_total: int = 0
    errors: int = 0
    warnings: int = 0
    by_layer: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    ok: bool
    xml_path: Optional[str]
    xml_digest: str
    catalog_path: Optional[str]
    xsd_path: Optional[str]
    summary: ValidationSummary
    issues: List[ValidationIssue]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "xml_path": self.xml_path,
            "xml_digest": self.xml_digest,
            "catalog_path": self.catalog_path,
            "xsd_path": self.xsd_path,
            "summary": self.summary.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ModelConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    device_map: str = "auto"
    dtype: str = "bf16"
    quantization: Optional[str] = None
    use_flash_attention: bool = False
    trust_remote_code: bool = False


@dataclass(slots=True)
class PeftConfig:
    method: Optional[str] = None
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    adapter_path: Optional[str] = None


@dataclass(slots=True)
class PromptConfig:
    mode: str = "zero_shot"
    few_shot_k: int = 0
    few_shot_dynamic: bool = True
    few_shot_pool_path: Optional[str] = None
    include_schema: bool = False
    include_xsd: bool = False
    xsd_max_chars: int = 8000
    system_rules: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TrainingConfig:
    output_dir: str = "artifacts"
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    do_sample: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    task: str
    output_root: str
    method: str
    model: ModelConfig
    peft: PeftConfig = field(default_factory=PeftConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    catalog_path: Optional[str] = None
    xsd_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunPaths:
    run_dir: Path
    mission_txt: Path
    experiment_json: Path
    prompt_rendered_txt: Path
    llm_output_raw_txt: Path
    generated_bt_xml: Path
    validation_report_json: Path
    metrics_json: Path
    summary_md: Path
    run_manifest_json: Path
