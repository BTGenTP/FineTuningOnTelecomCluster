"""
Grammaire NAV4RAIL pour décodage contraint
==========================================

Deux mécanismes de contrainte, complémentaires :

1. GBNF (llama.cpp / llama-cpp-python)
   ─────────────────────────────────────
   Format de grammaire BNF utilisé par llama.cpp pour contraindre la génération
   token par token. Utile si on bascule vers un backend llama.cpp (inférence CPU).

   Export : NAV4RAIL_GBNF  (str)

   Usage via llama-cpp-python :
       from llama_cpp import Llama, LlamaGrammar
       from nav4rail_grammar import NAV4RAIL_GBNF
       grammar = LlamaGrammar.from_string(NAV4RAIL_GBNF)
       llm = Llama(model_path="mistral-7b.gguf")
       out = llm("...", grammar=grammar)

2. lm-format-enforcer (HuggingFace Transformers + PEFT)
   ──────────────────────────────────────────────────────
   Contrainte basée sur un pattern regex, intégrée via prefix_allowed_tokens_fn.
   Compatible avec les modèles PEFT/LoRA chargés via transformers.

   Requiert : pip install lm-format-enforcer

   Export : get_prefix_fn(tokenizer)  → Callable

   Usage :
       from nav4rail_grammar import get_prefix_fn
       prefix_fn = get_prefix_fn(tokenizer)
       out = model.generate(**inputs, prefix_allowed_tokens_fn=prefix_fn, ...)

   Limitation connue : le regex ne peut pas vérifier l'équilibrage des balises
   (propriété context-sensitive). Il garantit en revanche :
     ✓ Uniquement des noms de tags du catalogue (zéro hallucination de skill)
     ✓ Structure <root BTCPP_format="4"> ... </root> correcte
     ✓ Attribut name= en snake_case
     ✗ Équilibrage parfait des balises (couvert par validate_bt.py en post-hoc)
"""

# ─── 1. Grammaire GBNF ────────────────────────────────────────────────────────
#
# Format : EBNF étendu utilisé par llama.cpp
# Référence : https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
#
# Conventions :
#   ::=   définition de règle
#   |     alternative
#   *     zéro ou plusieurs
#   +     un ou plusieurs
#   ?     optionnel
#   [...]  classe de caractères (comme regex)

NAV4RAIL_GBNF = r'''# ── NAV4RAIL Behavior Tree Grammar (GBNF) ──────────────────────────────────
# Génère du XML BehaviorTree.CPP v4 avec les 27 skills réels (4 familles).
# Utilisable avec llama.cpp (--grammar-file) ou llama-cpp-python.

root      ::= "<root BTCPP_format=\"4\">\n  <BehaviorTree ID=\"MainTree\">\n" node+ "  </BehaviorTree>\n</root>"

# Nœuds de contrôle (peuvent être imbriqués)
node      ::= sequence | fallback

sequence  ::= sp "<Sequence name=\"" name "\">\n" child+ sp "</Sequence>\n"
fallback  ::= sp "<Fallback name=\"" name "\">\n" child child child* sp "</Fallback>\n"

# Un enfant est soit un skill, soit un nœud de contrôle imbriqué
child     ::= skill | node

# Nœuds feuilles (skills — auto-fermants)
skill     ::= sp "<" skilltag " name=\"" name "\"/>\n"

# 27 skills réels NAV4RAIL (4 familles)
skilltag ::= "LoadMission" | "MissionStructureValid" | "UpdateCurrentGeneratedActivity" | "ProjectPointOnNetwork" | "CreatePath" | "AgregatePath" | "MissionFullyTreated" | "PassAdvancedPath" | "PassMission" | "GenerateMissionSequence" | "GenerateCorrectiveSubSequence" | "InsertCorrectiveSubSequence" | "MissionTerminated" | "CheckCurrentStepType" | "PassMotionParameters" | "Move" | "UpdateCurrentExecutedStep" | "Deccelerate" | "MoveAndStop" | "SignalAndWaitForOrder" | "IsRobotPoseProjectionActive" | "ManageMeasurements" | "AnalyseMeasurements" | "MeasurementsQualityValidated" | "PassDefectsLocalization" | "MeasurementsEnforcedValidated" | "SimulationStarted"

# Nom en snake_case
name      ::= [a-z] ([a-z0-9_])*

# Indentation flexible (2 à 10 espaces)
sp ::= "  " | "    " | "      " | "        " | "          "
'''


# ─── 2. Pattern regex pour lm-format-enforcer ────────────────────────────────

_SKILL = (
    "LoadMission|MissionStructureValid|UpdateCurrentGeneratedActivity"
    "|ProjectPointOnNetwork|CreatePath|AgregatePath|MissionFullyTreated"
    "|PassAdvancedPath|PassMission|GenerateMissionSequence"
    "|GenerateCorrectiveSubSequence|InsertCorrectiveSubSequence"
    "|MissionTerminated|CheckCurrentStepType|PassMotionParameters"
    "|Move|UpdateCurrentExecutedStep|Deccelerate|MoveAndStop"
    "|SignalAndWaitForOrder|IsRobotPoseProjectionActive"
    "|ManageMeasurements|AnalyseMeasurements|MeasurementsQualityValidated"
    "|PassDefectsLocalization|MeasurementsEnforcedValidated"
    "|SimulationStarted"
)
_CTRL = "Sequence|Fallback|Parallel"
_NAME = "[a-z][a-z0-9_]*"

# Skill auto-fermant :  <GetMission name="foo"/>
_SKILL_ELEM = fr'[ ]*<(?:{_SKILL}) name="{_NAME}"/>\n'

# Nœud de contrôle imbriqué (niveau 2) contenant des skills
_INNER_CTRL = (
    fr'[ ]*<(?:{_CTRL}) name="{_NAME}">\n'
    fr'(?:{_SKILL_ELEM})+'
    fr'[ ]*</(?:{_CTRL})>\n'
)

# Enfant d'un nœud de contrôle de niveau 1 : skill OU ctrl imbriqué
_CHILD = fr'(?:{_SKILL_ELEM}|{_INNER_CTRL})'

# Nœud de contrôle de niveau 1
_TOP_CTRL = (
    fr'[ ]*<(?:{_CTRL}) name="{_NAME}">\n'
    fr'(?:{_CHILD})+'
    fr'[ ]*</(?:{_CTRL})>\n'
)

# Pattern complet du BT (enveloppe root + BehaviorTree)
NAV4RAIL_XML_PATTERN = (
    r'<root BTCPP_format="4">\n'
    r'  <BehaviorTree ID="MainTree">\n'
    fr'(?:{_TOP_CTRL})+'
    r'  </BehaviorTree>\n'
    r'</root>'
)


# ─── 3. Helper HuggingFace ────────────────────────────────────────────────────

def get_prefix_fn(tokenizer):
    """
    Crée une prefix_allowed_tokens_fn pour model.generate() via lm-format-enforcer.

    À chaque étape de génération, cette fonction limite les tokens autorisés
    aux seuls tokens qui restent compatibles avec NAV4RAIL_XML_PATTERN.
    Cela garantit zéro hallucination de nom de skill.

    Args:
        tokenizer : tokenizer HuggingFace du modèle (Mistral, TinyLlama, etc.)

    Returns:
        Callable[[int, torch.Tensor], list[int]]
        → à passer à model.generate(prefix_allowed_tokens_fn=...)

    Raises:
        ImportError si lm-format-enforcer n'est pas installé.

    Exemple :
        prefix_fn = get_prefix_fn(tokenizer)
        out = model.generate(**inputs,
                             prefix_allowed_tokens_fn=prefix_fn,
                             max_new_tokens=600)
    """
    try:
        from lmformatenforcer import RegexParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError as exc:
        raise ImportError(
            "lm-format-enforcer non installé.\n"
            "Installez-le : pip install lm-format-enforcer\n"
            "Sur le cluster : ajoutez-le dans job_finetune.sh avant de soumettre."
        ) from exc

    parser = RegexParser(NAV4RAIL_XML_PATTERN)
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
