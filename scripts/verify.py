"""Offline pipeline check: tiny local BERT -> train on the starter split -> reload -> plan behaviour.

A randomly initialised tiny encoder is not expected to classify well, so grounding is
also checked with known-valid Table 1 combinations and the rule fallback's audit inputs.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_behavior.data import load_records  # noqa: E402
from context_behavior.dialogue import build_dialogue  # noqa: E402
from context_behavior.ontology import Ontology, check_table1  # noqa: E402
from context_behavior.rules import RuleInterpreter  # noqa: E402
from context_behavior.scene import BehaviorPlanner  # noqa: E402
from context_behavior.tiny import make_tiny_backbone  # noqa: E402

STARTER = ROOT / "src" / "context_behavior" / "resources" / "starter"


def run(*arguments: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run([sys.executable, *arguments], check=True, text=True, capture_output=True, env=environment)
    print(completed.stdout, end="")
    return completed.stdout


def main():
    out = ROOT / "outputs" / "verify"
    out.mkdir(parents=True, exist_ok=True)
    ontology = Ontology.load()
    check_table1(ontology)
    train, val = load_records(STARTER / "train.jsonl"), load_records(STARTER / "val.jsonl")
    backbone = make_tiny_backbone(out / "tiny-backbone", [r.text for r in train + val])
    model = out / "model"
    run("-m", "context_behavior.train", "--output", str(model), "--backbone", str(backbone), "--epochs", "3")
    for name in ("backbone/config.json", "tokenizer/vocab.txt", "ontology.json", "heads.pt", "metrics.json"):
        assert (model / name).is_file(), name
    scene = ROOT / "demo" / "scene.json"
    prediction = json.loads(run("-m", "context_behavior.infer", "--model", str(model), "--scene", str(scene), "Please sit on the chair"))
    for head in ("subject", "action", "position", "target"):
        assert prediction["prediction"][head] in ontology.classes[head]
    assert {"accepted", "route", "reason", "command", "steps"} <= set(prediction["dispatch"])
    (out / "prediction.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")

    planner = BehaviorPlanner.load(scene)
    sit = planner.plan({"subject": "Virtual Human", "action": "Sit", "position": "On", "target": "Chair"})
    assert sit.accepted and sit.command["behavior"] == "sit_on" and any(s["op"] == "sit" for s in sit.steps)
    rejected = planner.plan({"subject": "Virtual Human", "action": "Sit", "position": "In", "target": "Drawer"})
    assert not rejected.accepted and "does not support" in rejected.reason
    rules = RuleInterpreter(ontology)
    assert rules.predict("Don't open the window")["subject"] == "None"
    assert rules.predict("Turn the lamp off, it is on")["action"] == "Turn off"
    put = BehaviorPlanner.load(scene).plan(rules.predict("Put the pillow on the bed"))
    assert put.accepted and any(s["op"] == "place" and s["object"] == "pillow" for s in put.steps)
    assert build_dialogue("fixed").respond("hello")["reply"]
    (out / "known-valid-plans.json").write_text(json.dumps({"sit_on_chair": sit.to_dict(), "sit_in_drawer": rejected.to_dict(),
                                                            "put_pillow_on_bed": put.to_dict()}, indent=2), encoding="utf-8")
    print(f"verification passed: Table 1 ontology, starter data ({len(train)} train / {len(val)} val), tiny-BERT training, "
          f"self-contained checkpoint reload, behaviour planning and rule regressions -> {out}")


if __name__ == "__main__":
    main()
