"""
Prompt builder for NAV4RAIL benchmarking.
==========================================
Constructs prompts in multiple modes for different evaluation strategies.

Modes:
  - zero_shot: system prompt + mission only
  - few_shot: system prompt + k examples + mission
  - schema_guided: system prompt + catalog excerpt + mission
  - chain_of_thought: system prompt + reasoning template + mission
  - sft: formatted for training (system/user/assistant roles)
  - program_of_thoughts: Code-as-Reasoning — ask for a Python script against
    MissionBuilder API. The agent executes the script in a sandbox.
  - react_agent: Same as PoT but supports an error `history` for iterative
    refinement (used by the LangGraph ReAct loop).

Reuses SYSTEM_PROMPT from finetune/finetune_llama3_nav4rail.py.
"""

from __future__ import annotations

from typing import Any

from src.data.skills_loader import SafetyRulesLoader, SkillsCatalog

# ── System Prompt (from finetune/finetune_llama3_nav4rail.py lines 41-117) ──

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

Reponds uniquement avec le XML."""


# ── Few-shot examples ────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "category": "transport",
        "mission": "Mission de transport simple : deplacer le robot du point A au point B sur la voie 3, avec arret intermediaire au PK 125.3 pour autorisation.",
        "xml": """\
<root BTCPP_format="4" main_tree_to_execute="transport_mission">
  <BehaviorTree ID="transport_mission">
    <Sequence name="TRANSPORT MISSION">
      <SubTreePlus name="PREPARATION" ID="base_preparation" __autoremap="true"/>
      <SubTreePlus name="EXECUTE" ID="execute" __autoremap="true"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="base_preparation">
    <Sequence name="BASE PREPARATION">
      <SubTreePlus name="GET MISSION" ID="get_mission" __autoremap="true"/>
      <SubTreePlus name="CALCULATE PATH" ID="calculate_path" __autoremap="true"/>
      <Action name="PASS ADVANCED PATH" ID="PassAdvancedPath" adv_path="{adv_path}"/>
      <Action name="PASS MISSION" ID="PassMission" mission="{mission}"/>
      <Action name="GENERATE MISSION SEQUENCE" ID="GenerateMissionSequence" mission="{mission}" mission_sequence="{mission_sequence}"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="get_mission">
    <Sequence name="GET MISSION">
      <Action name="LOAD MISSION" ID="LoadMission" mission_file_path="{mission_file_path}"/>
      <Condition name="MISSION STRUCTURE VALID" ID="MissionStructureValid"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="calculate_path">
    <Fallback name="PATH CALCULATION">
      <Repeat name="LOOP" num_cycles="-1">
        <Sequence name="ACTIVITY">
          <Action name="UPDATE ACTIVITY" ID="UpdateCurrentGeneratedActivity" type="{type}" origin_sph="{origin_sph}" target_sph="{target_sph}" forbidden_atoms_out="{forbidden_atoms}"/>
          <Action name="PROJECT ORIGIN" ID="ProjectPointOnNetwork" point_in="{origin_sph}" point_out="{origin}"/>
          <Action name="PROJECT TARGET" ID="ProjectPointOnNetwork" point_in="{target_sph}" point_out="{target}"/>
          <Action name="CREATE PATH" ID="CreatePath" origin="{origin}" target="{target}" forbidden_atoms="{forbidden_atoms}" path="{path}"/>
          <Action name="AGREGATE PATH" ID="AgregatePath" path="{path}"/>
        </Sequence>
      </Repeat>
      <Condition name="MISSION FULLY TREATED" ID="MissionFullyTreated" type="{type}"/>
    </Fallback>
  </BehaviorTree>
  <BehaviorTree ID="execute">
    <ReactiveFallback name="EXECUTE">
      <Repeat name="STEP LOOP" num_cycles="-1">
        <Fallback name="MOTION SELECTOR">
          <SubTreePlus name="MOVE" ID="move" __autoremap="true"/>
          <SubTreePlus name="DECCELERATE" ID="deccelerate" __autoremap="true"/>
          <SubTreePlus name="REACH AND STOP" ID="reach_and_stop" __autoremap="true"/>
          <SubTreePlus name="PASS" ID="pass" __autoremap="true"/>
          <SubTreePlus name="REACH STOP NO WAIT" ID="reach_stop_no_wait" __autoremap="true"/>
        </Fallback>
      </Repeat>
      <Condition name="MISSION TERMINATED" ID="MissionTerminated"/>
    </ReactiveFallback>
  </BehaviorTree>
  <BehaviorTree ID="move">
    <Sequence name="MOVE">
      <Condition name="IS MOVE" ID="CheckCurrentStepType" type_to_be_checked="0"/>
      <Action name="PASS PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="MOVE" ID="Move" threshold_type="1" motion_params="{motion_params}"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="deccelerate">
    <Sequence name="DECCELERATE">
      <Condition name="IS DECEL" ID="CheckCurrentStepType" type_to_be_checked="1"/>
      <Action name="PASS PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="DECCELERATE" ID="Deccelerate" motion_params="{motion_params}"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="reach_and_stop">
    <Sequence name="REACH AND STOP">
      <Condition name="IS REACH STOP" ID="CheckCurrentStepType" type_to_be_checked="2"/>
      <Action name="PASS PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="MOVE AND STOP" ID="MoveAndStop" motion_params="{motion_params}"/>
      <Action name="SIGNAL" ID="SignalAndWaitForOrder" message="autorisation requise PK 125.3"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="pass">
    <Sequence name="PASS">
      <Condition name="IS PASS" ID="CheckCurrentStepType" type_to_be_checked="3"/>
      <Action name="PASS PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="MOVE" ID="Move" threshold_type="3" motion_params="{motion_params}"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
  <BehaviorTree ID="reach_stop_no_wait">
    <Sequence name="REACH STOP NO WAIT">
      <Condition name="IS REACH STOP NO WAIT" ID="CheckCurrentStepType" type_to_be_checked="4"/>
      <Action name="PASS PARAMS" ID="PassMotionParameters" motion_params="{motion_params}"/>
      <Action name="MOVE AND STOP" ID="MoveAndStop" motion_params="{motion_params}"/>
      <Action name="UPDATE STEP" ID="UpdateCurrentExecutedStep"/>
    </Sequence>
  </BehaviorTree>
</root>""",
    },
]


def build_system_prompt(safety_rules: SafetyRulesLoader | None = None) -> str:
    """
    NAV4RAIL XML system string (plus optional safety rules).

    PoT / ReAct eval uses :data:`CODE_SYSTEM_PROMPT` instead; see
    :func:`system_message_body_for_mode` for the string that matches
    ``build_prompt`` per mode.
    """

    system_content = SYSTEM_PROMPT
    if safety_rules:
        system_content += "\n\n" + safety_rules.summarize_for_prompt()
    return system_content


def system_message_body_for_mode(
    mode: str,
    safety_rules: SafetyRulesLoader | None = None,
) -> str:
    """
    System-role text exactly as ``build_prompt`` chooses for ``mode``.

    Accepts benchmark ``training.method`` values (``pot``) and internal
    ``build_prompt`` names (``program_of_thoughts``).
    """

    if mode in ("pot", "react_agent", "program_of_thoughts"):
        text = CODE_SYSTEM_PROMPT
        if safety_rules is not None:
            text += "\n\n" + safety_rules.summarize_for_prompt()
        return text
    return build_system_prompt(safety_rules=safety_rules)


# ── Chain-of-Thought Template ────────────────────────────────────────────────

COT_TEMPLATE = """\
Analyse la mission suivante etape par etape avant de generer le BT XML :

1. TYPE DE MISSION : Transport simple, transport avec autorisation, inspection avec controle, inspection sans controle, ou mission complexe ?
2. SUBTREES NECESSAIRES : Quels motion subtrees inclure ? (types 0-4 pour transport, 10-14 pour inspection)
3. INSPECTION : Si inspection, faut-il AnalyseMeasurements + correction (types 12-14 avec controle) ?
4. SECURITE : Verifier que LoadMission est premier, MoveAndStop est present, execution dans ReactiveFallback(Repeat(-1), MissionTerminated)

Mission : {mission}

Raisonnement :
"""


# ── Code-as-Reasoning prompts (PoT / ReAct) ──────────────────────────────────

CODE_SYSTEM_PROMPT = """\
Tu es un expert en robotique ferroviaire NAV4RAIL. Pour chaque mission, tu ecris un
SCRIPT PYTHON qui utilise l'API MissionBuilder pour construire un BehaviorTree XML
BehaviorTree.CPP v4. Le script est execute dans un sandbox restreint et son stdout
(ou la variable `xml`) devient le BT genere.

REGLES :
- Reponds UNIQUEMENT avec un bloc de code Python dans une cloture ```python ... ```.
- Le script DOIT se terminer par `print(builder.to_xml())` (ou affecter `xml = builder.to_xml()`).
- Seul le module `nav4rail_builder` peut etre importe. Aucune I/O, aucun os/subprocess.
- Toute erreur levee par MissionBuilder (UnknownSkillError, PortError, StructuralError,
  MissingRequiredSkillError) indique une mauvaise construction — corrige la logique.
- Prefere les helpers haut-niveau (`add_get_mission`, `add_calculate_path`,
  `add_base_preparation`, `add_execute`, `add_main_tree`) qui encodent les regles SR-023..SR-027.
- Choisis les `step_types` en fonction de la mission :
    Transport : inclure au moins un type parmi 0,1,2,3,4.
    Inspection SANS controle : ajouter 10/11 (et 13/14 si non-arret) avec mesures seulement.
    Inspection AVEC controle : ajouter 12 (requiert AnalyseMeasurements + Fallback correctif)
    et eventuellement 13/14.
"""

POT_TEMPLATE = """\
{api_docs}

Mission : {mission}

Ecris le script Python (dans un bloc ```python ... ```) qui construit le BehaviorTree
pour cette mission et affiche le XML final via `print(builder.to_xml())`.
"""

REACT_INITIAL_TEMPLATE = POT_TEMPLATE

REACT_REFINE_TEMPLATE = """\
{api_docs}

Mission : {mission}

Tentative precedente :
```python
{previous_code}
```

Erreur rencontree (tour {iteration}) :
{error_feedback}

{validator_feedback}

Corrige le script pour eliminer l'erreur ci-dessus. Reponds UNIQUEMENT avec un nouveau
bloc ```python ... ``` complet (pas de diff). Termine par `print(builder.to_xml())`.
"""


def _format_history_feedback(history: list[dict[str, Any]] | None) -> tuple[str, str, str, int]:
    """
    Extract the most recent attempt's code + errors from an iteration history.

    Each history entry is expected to have keys: code, error, validator (optional).
    Returns (previous_code, error_feedback, validator_feedback, iteration_index).
    If history is empty, returns ("", "", "", 0).
    """
    if not history:
        return "", "", "", 0

    last = history[-1]
    previous_code = str(last.get("code", "")).strip()
    error = last.get("error") or "(aucune exception — mais le score de validation est < 1.0)"
    validator = last.get("validator") or ""
    iteration = int(last.get("iteration", len(history)))

    if validator:
        validator_section = f"Validator feedback :\n{validator}"
    else:
        validator_section = ""

    return previous_code, str(error), validator_section, iteration


# ── Prompt Builder ───────────────────────────────────────────────────────────


def build_prompt(
    mode: str,
    mission: str,
    model_config: dict | None = None,
    catalog: SkillsCatalog | None = None,
    safety_rules: SafetyRulesLoader | None = None,
    k_examples: int = 1,
    history: list[dict[str, Any]] | None = None,
) -> str | list[dict[str, str]]:
    """
    Build prompt for a given mode.

    Args:
        mode: "zero_shot", "few_shot", "schema_guided", "chain_of_thought",
              "sft", "program_of_thoughts", "react_agent"
        mission: Natural language mission description
        model_config: Model-specific config (from base.yaml models section)
        catalog: SkillsCatalog for schema injection
        safety_rules: SafetyRulesLoader for rule injection
        k_examples: Number of few-shot examples (for "few_shot" mode)
        history: List of prior attempts for "react_agent" mode. Each entry is a
                 dict with keys code, error, validator, iteration. If non-empty,
                 the prompt becomes a refinement request.

    Returns:
        For chat_template models: list of dicts [{"role": ..., "content": ...}]
        For Mistral: formatted string with [INST]...[/INST]
    """
    model_config = model_config or {}
    use_chat = model_config.get("chat_template", True)
    supports_system = model_config.get("supports_system", True)

    # Code-as-Reasoning modes override the default XML-focused system prompt.
    if mode in {"program_of_thoughts", "react_agent"}:
        system_content = CODE_SYSTEM_PROMPT
        if safety_rules is not None:
            system_content += "\n\n" + safety_rules.summarize_for_prompt()
    else:
        system_content = build_system_prompt(safety_rules=safety_rules)

    if mode == "zero_shot":
        user_content = f"Mission : {mission}"

    elif mode == "few_shot":
        examples = FEW_SHOT_EXAMPLES[:k_examples]

    elif mode == "schema_guided":
        if catalog:
            schema = catalog.summarize()
        else:
            schema = "(catalog non disponible)"
        user_content = f"Schema des skills disponibles :\n{schema}\n\nMission : {mission}"

    elif mode == "chain_of_thought":
        user_content = COT_TEMPLATE.format(mission=mission)

    elif mode == "sft":
        user_content = f"Mission : {mission}"

    elif mode == "program_of_thoughts":
        from src.builder.api_docs import get_full_api_docs

        api_docs = get_full_api_docs(catalog)
        user_content = POT_TEMPLATE.format(api_docs=api_docs, mission=mission)

    elif mode == "react_agent":
        from src.builder.api_docs import get_full_api_docs

        api_docs = get_full_api_docs(catalog)
        if history:
            prev_code, error_feedback, validator_feedback, iteration = (
                _format_history_feedback(history)
            )
            user_content = REACT_REFINE_TEMPLATE.format(
                api_docs=api_docs,
                mission=mission,
                previous_code=prev_code,
                error_feedback=error_feedback,
                validator_feedback=validator_feedback,
                iteration=iteration,
            )
        else:
            user_content = REACT_INITIAL_TEMPLATE.format(
                api_docs=api_docs, mission=mission
            )

    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    # Format based on model type
    if use_chat:
        messages: list[dict[str, str]] = []

        if mode == "few_shot":
            if supports_system:
                messages.append({"role": "system", "content": system_content})
                prefix = ""
            else:
                # Models without a system role (e.g. Gemma): prepend once to the first user turn.
                prefix = system_content + "\n\n"

            for j, ex in enumerate(examples):
                user_turn = f"Mission : {ex['mission']}"
                if j == 0 and prefix:
                    user_turn = prefix + user_turn
                messages.append({"role": "user", "content": user_turn})
                messages.append({"role": "assistant", "content": ex["xml"]})

            final_user = f"Mission : {mission}"
            if not examples and prefix:
                final_user = prefix + final_user
            messages.append({"role": "user", "content": final_user})
            return messages

        if supports_system:
            messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": user_content})
        else:
            # Gemma: no system role, prepend to user message
            messages.append({"role": "user", "content": system_content + "\n\n" + user_content})
        return messages
    else:
        # Mistral [INST]...[/INST] format
        if mode == "few_shot":
            turns: list[str] = []
            for j, ex in enumerate(examples):
                user_turn = f"Mission : {ex['mission']}"
                if j == 0:
                    user_turn = system_content + "\n\n" + user_turn
                turns.append(f"<s>[INST] {user_turn} [/INST] {ex['xml']} </s>")

            final_user = f"Mission : {mission}"
            if not examples:
                final_user = system_content + "\n\n" + final_user
            turns.append(f"<s>[INST] {final_user} [/INST]")
            return "\n".join(turns)

        return f"[INST] {system_content}\n\n{user_content} [/INST]"


def build_sft_example(
    mission: str,
    xml_response: str,
    model_config: dict | None = None,
    catalog: SkillsCatalog | None = None,
) -> str | list[dict[str, str]]:
    """
    Build a full SFT training example (prompt + response).
    For chat template models: returns messages with assistant response.
    """
    messages = build_prompt("sft", mission, model_config, catalog)
    if isinstance(messages, list):
        messages.append({"role": "assistant", "content": xml_response})
        return messages
    else:
        return messages + " " + xml_response
