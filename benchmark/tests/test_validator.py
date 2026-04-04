from __future__ import annotations

import sys
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.data.synthetic_generator import build_nav4rail_tree
from src.rewards.validator import validate


CATALOG_PATH = BENCHMARK_ROOT / "data" / "nav4rail_catalog.json"


def test_validator_accepts_valid_nav4rail_tree() -> None:
    xml = build_nav4rail_tree()
    report = validate(xml_text=xml, catalog_path=CATALOG_PATH, strict=True)
    assert report.ok, report.to_dict()
    assert report.summary.errors == 0


def test_validator_rejects_malformed_xml() -> None:
    report = validate(xml_text="<root><BehaviorTree></root>", catalog_path=CATALOG_PATH, strict=True)
    assert not report.ok
    assert any(issue.code == "xml_parse_error" for issue in report.issues)


def test_validator_rejects_unknown_skill_and_missing_attr() -> None:
    xml = """
    <root main_tree_to_execute="MainTree">
      <BehaviorTree ID="MainTree">
        <Sequence name="bad">
          <Action ID="Move" threshold_type="1"/>
          <Action ID="NotInCatalog"/>
        </Sequence>
      </BehaviorTree>
    </root>
    """
    report = validate(xml_text=xml, catalog_path=CATALOG_PATH, strict=True)
    assert not report.ok
    codes = {issue.code for issue in report.issues}
    assert "missing_skill_attr" in codes
    assert "skill_not_allowed" in codes


def test_validator_rejects_blackboard_chain_violation() -> None:
    xml = """
    <root main_tree_to_execute="MainTree">
      <BehaviorTree ID="MainTree">
        <Sequence name="bad_chain">
          <Action ID="Move" name="MOVE" threshold_type="1" motion_params="{motion_params}"/>
          <Action ID="PassMotionParameters" name="PASS" motion_params="{motion_params}"/>
        </Sequence>
      </BehaviorTree>
      <BehaviorTree ID="execute">
        <ReactiveFallback name="EXECUTE">
          <Repeat name="loop" num_cycles="-1">
            <Fallback name="selector">
              <SubTreePlus ID="MainTree"/>
            </Fallback>
          </Repeat>
          <Condition ID="MissionTerminated"/>
        </ReactiveFallback>
      </BehaviorTree>
    </root>
    """
    report = validate(xml_text=xml, catalog_path=CATALOG_PATH, strict=True)
    assert not report.ok
    codes = {issue.code for issue in report.issues}
    assert "blackboard_unproduced" in codes
    assert "nav4rail_blackboard_chain" in codes


def test_validator_rejects_inspection_without_analysis() -> None:
    xml = """
    <root main_tree_to_execute="MainTree">
      <BehaviorTree ID="MainTree">
        <ReactiveFallback name="EXECUTE">
          <Repeat name="loop" num_cycles="-1">
            <Fallback name="selector">
              <SubTreePlus ID="inspect_step"/>
            </Fallback>
          </Repeat>
          <Condition ID="MissionTerminated"/>
        </ReactiveFallback>
      </BehaviorTree>
      <BehaviorTree ID="inspect_step">
        <Sequence name="inspect">
          <Condition ID="CheckCurrentStepType" type_to_be_checked="10"/>
          <Action ID="PassMotionParameters" motion_params="{motion_params}"/>
          <Action ID="Move" threshold_type="1" motion_params="{motion_params}"/>
          <Action ID="UpdateCurrentExecutedStep"/>
        </Sequence>
      </BehaviorTree>
    </root>
    """
    report = validate(xml_text=xml, catalog_path=CATALOG_PATH, strict=True)
    assert not report.ok
    codes = {issue.code for issue in report.issues}
    assert "inspection_requires_analysis" in codes
