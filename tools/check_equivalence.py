#!/usr/bin/env python3
"""Compare installed frozen code with small, source-derived CPU golden records.

Maintainers may pass two read-only Git roots to independently recreate the
reference. Neither mode imports a source worktree or starts external services.
"""
from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/equivalence"
REVISIONS = {"bench": "5b8d2b37fd7d97b161e88de4d438ec789c78f9b4",
             "runtime": "9b992b6797330a836e9533e5409627bf63f7d94b"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def normalize(value, roots):
    """Only relocate roots and compare decoded image pixels across Pillow builds."""
    if isinstance(value, Mapping):
        return {str(key): normalize(item, roots) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item, roots) for item in value]
    if isinstance(value, str):
        if value.startswith("data:image/png;base64,"):
            from PIL import Image
            with Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))) as image:
                return {"decoded_image_mode": image.mode, "size": list(image.size),
                        "pixels_sha256": sha(image.tobytes())}
        for root, label in roots:
            value = value.replace(str(root), label)
        return value
    return value


def message(arguments, **fields):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "fixture_call", "type": "function", "function": {
            "name": "robot_action", "arguments": json.dumps(arguments)}}], **fields}


def completion(arguments):
    return {"choices": [{"finish_reason": "tool_calls", "message": message(arguments)}]}


def outcome(call):
    try:
        return {"value": call()}
    except (ValueError, RuntimeError, TypeError) as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)}


def probe(layout, work):
    if layout:
        roots = {name + "_root": layout / name for name in REVISIONS}
        sys.path[:0] = [str(roots["bench_root"]), str(roots["bench_root"] / "lab"),
                        str(roots["runtime_root"] / "src"), str(roots["runtime_root"] / "scripts")]
    else:
        from esi_benchmark.bootstrap import activate
        roots = activate()
    from Agent import (EpisodeConversation, PrivateEpisodeContract, PublicImage,
                       PublicObservation, PublicToolResult, load_common_system_prompt_revision)
    from agent_protocol import parse_native_tool_message
    from agent_observation import (approximate_depth_at_anchor, marker_depth_records,
                                   model_visible_marker_depth_text, validate_public_marker_depth)
    from behavior_hl.embodied.evaluator_bridge.official_predicate_verifier import OfficialPredicateVerifier
    from behavior_hl.embodied.evaluator_bridge.l0_verifier_adapter import ProjectedGoalVerifier
    from behavior_hl.embodied.evaluator_bridge.l1_process_audit import ProcessAudit
    from PIL import Image

    fixture = json.loads((FIXTURES / "inputs.json").read_text())
    values = {}
    root_labels = [(work, "$WORK"), (roots["bench_root"], "$BENCH"),
                   (roots["runtime_root"], "$RUNTIME")]

    def save(name, value):
        values[name] = normalize(value, root_labels)

    def observation(name):
        directory = work / "observations" / name
        directory.mkdir(parents=True)
        images = []
        for camera, color, edge in (("head", (0, 0, 255), 4),
                                    ("left_wrist", (255, 0, 0), 2),
                                    ("right_wrist", (0, 255, 0), 2)):
            path = directory / (camera + ".png")
            Image.new("RGB", (edge, edge), color).save(path)
            images.append(PublicImage(camera, path))
        return PublicObservation(name, tuple(images))

    def contract(name):
        return PrivateEpisodeContract(name, "synthetic-task", "synthetic", "0", 7,
                                      "recorded-provider", work, "Move the visible cup.", 5)

    for case in fixture["parser"]:
        def parse(case=case):
            raw = message(case["arguments"], **case.get("message_fields", {}))
            if "argument_text" in case:
                raw["tool_calls"][0]["function"]["arguments"] = case["argument_text"]
            result = parse_native_tool_message(raw, **case.get("options", {}))
            return {**result.__dict__, "function_arguments": result.function_arguments(),
                    "assistant_message": result.assistant_message(),
                    "physical_repeat": result.physical_repeat}
        save("parser/" + case["id"], outcome(parse))

    for case in fixture["public_results"]:
        save("result/" + case["id"], outcome(lambda case=case:
             PublicToolResult(**case["fields"]).to_public_dict()))

    for case in fixture["depth"]:
        save("observation/depth_" + case["id"], outcome(lambda case=case:
             approximate_depth_at_anchor(case["array"], tuple(case["anchor"]), radius=1)))
    depth = marker_depth_records(head_camera={"depth_linear": [[1, 2], [3, 4]]},
                                 anchors={"C2": (1, 1), "C1": (0, 0)})
    save("observation/marker_depth", {"records": depth, "text": model_visible_marker_depth_text(depth)})
    save("observation/private_field_rejected", outcome(lambda:
         validate_public_marker_depth([{**depth[0], "object_id": "private-object"}])))

    conversation = EpisodeConversation(contract("marker-history"), observation("initial"))
    save("conversation/initial_request", conversation.build_native_request())
    for index, step in enumerate(fixture["conversation"]):
        dispatch = conversation.append_assistant_completion(completion(step["arguments"]))
        expected_stage = conversation.expected_selection_stage
        conversation.append_tool_result(PublicToolResult(**step["result"]))
        conversation.append_observation(observation("step_" + str(index)), kind=step["kind"],
                                        marker_targets=step.get("targets", ()))
        save("conversation/step_" + str(index), {
            "runtime_action": dict(dispatch.runtime_action), "expected_stage": expected_stage,
            "pending_stage": conversation.pending_marker_stage,
            "pending_targets": sorted(conversation.pending_marker_targets),
            "active_observations": conversation.active_observation_ids,
            "macro_actions": conversation.macro_action_count,
            "request": conversation.build_native_request(preserve_reasoning=True)})
    transcript = conversation.artifact_transcript(artifact_root=work)
    save("trajectory/transcript", transcript)
    save("trajectory/selected_evidence", conversation.artifact_selected_target_visual_evidence())
    save("trajectory/chatml", conversation.artifact_chatml(artifact_root=work))
    restored = EpisodeConversation.from_artifact_transcript(
        transcript, expected_contract=contract("marker-history"),
        system_prompt_revision=load_common_system_prompt_revision(physical_history_max=5),
        artifact_root=work)
    save("trajectory/restored_request", restored.build_native_request())
    restored.append_assistant_completion(completion({"action": "done"}))
    restored.append_tool_result(PublicToolResult("completed"))
    restored.finish_after_done()
    save("termination/done", restored.artifact_transcript(artifact_root=work))

    cancel = EpisodeConversation(contract("cancel"), observation("cancel_initial"))
    cancel.append_assistant_completion(completion({"action": "pick"}))
    cancel.append_tool_result(PublicToolResult("selection_required", stage=1))
    cancel.append_observation(observation("cancel_markers"), kind="marker", marker_targets=(1,))
    before = canonical(normalize(cancel.artifact_transcript(), root_labels))
    rejected = outcome(lambda: dict(cancel.append_assistant_completion(
        completion({"action": "wait"}), remaining_action_decisions=1).runtime_action))
    save("budget/cancel_rejected", {"result": rejected,
         "state_unchanged": before == canonical(normalize(cancel.artifact_transcript(), root_labels))})
    dispatch = cancel.append_assistant_completion(completion({"action": "wait"}), remaining_action_decisions=2)
    cancel.append_tool_result(PublicToolResult("completed", action_decisions_remaining=0))
    cancel.append_observation(observation("cancel_finished"), kind="physical")
    save("budget/cancel_allowed", {"dispatch": dict(dispatch.runtime_action),
         "macro_actions": cancel.macro_action_count, "request": cancel.build_native_request()})
    cancel.finish_after_official_success()
    save("termination/official_success", cancel.artifact_transcript(artifact_root=work))

    official = OfficialPredicateVerifier()
    for case in fixture["official_scores"]:
        save("score/official_" + case["id"], official._score_options(
            tuple(map(tuple, case["current"])), tuple(map(tuple, case["initial"]))))
    for case in fixture["projected_scores"]:
        def reader(truth):
            return lambda predicate, target, destination: truth[target]
        verifier = ProjectedGoalVerifier(case["goal"], reader(case["initial"]),
                                         stability_window=case.get("stability_window", 1))
        reports = [verifier.snapshot(reader(step["truth"]), done_requested=step.get("done", False))
                   for step in case["steps"]]
        save("score/projected_" + case["id"], reports)
    for case in fixture["process"]:
        audit = ProcessAudit(case["requirements"], role_object=lambda role: role,
                             recipe_truth=lambda *args: True, protected_relations=[],
                             directory=work / case["id"])
        reports = []
        for index, step in enumerate(case["steps"], 1):
            audit.observe(**step, action_index=index)
            reports.append(audit.report())
        save("process/" + case["id"], reports)

    # The production model loop consumes records; it never contacts a provider or scene.
    import model_client
    root_labels.append((model_client.PROJECT_ROOT, "$RUNTIME"))
    for case in fixture["model_loop"]:
        directory = work / case["id"]
        directory.mkdir()
        source_observation = observation(case["id"] + "_source")
        images = [{**image.artifact_record(), "sha256": sha(image.path.read_bytes())}
                  for image in source_observation.images]
        requests, commands = [], []

        class Provider:
            model = "recorded-provider"
            maximum_output_tokens = None
            cursor = 0

            def capability_preflight(self, *, artifact, **kwargs):
                report = {"status": "PASS", "simulator_touched": False}
                model_client._write_json(artifact, report)
                return report

            def complete(self, request, *, call_dir, seed):
                requests.append({"seed": seed, "request": request})
                call_dir.mkdir(parents=True)
                record = case["responses"][self.cursor]
                self.cursor += 1
                response = completion(record) if "action" in record else record
                model_client._write_json(call_dir / "response.json", response)
                return response

        class Scene:
            scene_run_dir = source_observation.images[0].path.parent
            cursor = 0

            def request(self, command, **fields):
                commands.append({"command": command, **fields})
                if command == "status":
                    return {"scene_resident": True, "session_valid": True,
                            "episode_active": False, "episode_id": None, "next_sequence": 0}
                if command == "begin_episode":
                    return {"observation": {"observation_id": "initial", "kind": "physical",
                            "images": images, "marker_targets": []}}
                if command == "end_episode":
                    return {"status": "PASS"}
                if command != "step":
                    raise AssertionError("Unexpected fixture scene command")
                step = case["steps"][self.cursor]
                self.cursor += 1
                return {"action_result": step["result"], "observation": {
                    "observation_id": "loop_" + str(self.cursor), "kind": step["kind"],
                    "images": images, "marker_targets": step.get("targets", [])},
                    **step.get("extra", {})}

        summary = model_client.run_model_episode(
            scene=Scene(), provider=Provider(), experiment_id=case["id"],
            instruction="Synthetic protocol fixture.", physical_history_max=5,
            maximum_model_calls=case.get("maximum_model_calls", 10),
            maximum_action_decisions=case.get("maximum_action_decisions", 150),
            maximum_consecutive_blocked=case.get("maximum_consecutive_blocked", 30),
            output_dir=directory)
        save("model_loop/" + case["id"], {"summary": summary, "requests": requests,
             "commands": commands, "transcript": json.loads((directory / "conversation_private.json").read_text())})

    files = {}
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        for name in REVISIONS:
            if path.is_relative_to(roots[name + "_root"]):
                relative = path.relative_to(roots[name + "_root"]).as_posix()
                files[name + "/" + relative] = sha(path.read_bytes())
    for name in ("action20_core_instructions.txt", "action20_tool_schema.json", "action20_bare_marker_schema.json"):
        files["bench/" + name] = sha((roots["bench_root"] / name).read_bytes())
    return {"cases": values, "loaded_source_sha256": files}


def run_probe(layout, work, output):
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", str(output), "--work", str(work)]
    if layout:
        command += ["--layout", str(layout)]
    subprocess.run(command, check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", nargs=2, metavar=("BENCH_GIT", "RUNTIME_GIT"))
    parser.add_argument("--refresh-golden", action="store_true", help="Requires frozen read-only source roots")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--probe", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--layout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        args.work = args.work.resolve()
        args.work.mkdir(parents=True)
        args.probe.write_text(json.dumps(probe(args.layout, args.work), indent=2, ensure_ascii=False) + "\n")
        return 0
    if args.refresh_golden and not args.source_roots:
        parser.error("--refresh-golden requires --source-roots")
    temporary = ROOT / ".tmp"
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="equivalence-", dir=temporary) as temp:
        temp = Path(temp)
        current = run_probe(None, temp / "candidate-work", temp / "candidate.json")
        if args.source_roots:
            source_roots = dict(zip(REVISIONS, map(Path, args.source_roots), strict=True))
            layout = temp / "reference"
            for key in current["loaded_source_sha256"]:
                component, relative = key.split("/", 1)
                raw = subprocess.run(["git", "-C", str(source_roots[component]), "show",
                                      REVISIONS[component] + ":" + relative], check=True, capture_output=True).stdout
                destination = layout / key
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
            reference = run_probe(layout, temp / "reference-work", temp / "reference.json")
        else:
            reference = json.loads((FIXTURES / "golden.json").read_text())
        if args.refresh_golden:
            (FIXTURES / "golden.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")
        keys = sorted(set(reference["cases"]) | set(current["cases"]))
        differences = [key for key in keys if reference["cases"].get(key) != current["cases"].get(key)]
        groups = {}
        for key in keys:
            group = key.split("/", 1)[0]
            groups[group] = groups.get(group, 0) + 1
        report = {"schema_version": "esi-benchmark/behavior-equivalence-v1",
                  "status": "PASS" if not differences else "FAIL", "case_count": len(keys),
                  "case_groups": groups, "differences": differences, "source_revisions": REVISIONS,
                  "reference_mode": "frozen_git_subprocess" if args.source_roots else "frozen_source_golden",
                  "evidence_scope": "CPU fixed-input comparison; synthetic scene/provider; no Live verification",
                  "normalization": ["temporary, installed and upstream configured project roots replaced by stable labels",
                                    "PNG data URLs compared as mode, size and decoded pixel SHA256"],
                  "inputs_sha256": sha((FIXTURES / "inputs.json").read_bytes()),
                  "reference_output_sha256": sha(canonical(reference["cases"])),
                  "candidate_output_sha256": sha(canonical(current["cases"])),
                  "reference_source_sha256": reference["loaded_source_sha256"],
                  "candidate_source_sha256": current["loaded_source_sha256"]}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "case_count", "case_groups", "differences")}))
        return int(bool(differences))


if __name__ == "__main__":
    raise SystemExit(main())
