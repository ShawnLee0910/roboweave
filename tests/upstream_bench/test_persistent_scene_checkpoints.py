from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pytest

from lab.persistent_scene_checkpoints import (
    PersistentCheckpointError,
    decode_portable_value,
    encode_portable_value,
    load_persistent_checkpoint,
    write_persistent_checkpoint,
)


@dataclass(frozen=True)
class _KnownRecord:
    name: str
    payload: object


class _Object:
    def __init__(self, name: str) -> None:
        self.name = name


def _encode_custom(value):
    if isinstance(value, _KnownRecord):
        return "known_record", {"name": value.name, "payload": value.payload}
    return None


def _decode_custom(tag, fields):
    if tag != "known_record" or set(fields) != {"name", "payload"}:
        raise PersistentCheckpointError("unknown test dataclass")
    return _KnownRecord(name=fields["name"], payload=fields["payload"])


def test_portable_value_round_trip_rebinds_objects_and_arrays() -> None:
    source = _Object("cup_1")
    replacement = _Object("cup_1")
    value = {
        "held": source,
        "pose": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "nested": _KnownRecord("inventory", (source, frozenset({1, 2}))),
    }
    encoded = encode_portable_value(
        value,
        object_name_for=lambda item: item.name if item is source else None,
        dataclass_encoder=_encode_custom,
    )
    decoded = decode_portable_value(
        encoded,
        object_for_name=lambda name: replacement if name == "cup_1" else None,
        dataclass_decoder=_decode_custom,
    )
    assert decoded["held"] is replacement
    assert np.array_equal(decoded["pose"], value["pose"])
    assert decoded["nested"].payload[0] is replacement
    assert decoded["nested"].payload[1] == frozenset({1, 2})


def test_portable_value_rejects_unknown_executable_object() -> None:
    with pytest.raises(PersistentCheckpointError, match="unsupported portable"):
        encode_portable_value(
            object(),
            object_name_for=lambda _value: None,
            dataclass_encoder=lambda _value: None,
        )


def test_write_load_and_prune_two_persistent_checkpoints(tmp_path) -> None:
    identity = {
        "task_id": 0,
        "task_name": "turning_on_radio",
        "task_split": "train",
        "task_instance_id": "0",
        "task_seed": 19506,
        "scene_model": "house_single_floor",
        "scene_load_mode": "full",
    }
    records = []
    for action_count, retained in ((50, [50]), (100, [50, 100]), (150, [100, 150])):
        records.append(
            write_persistent_checkpoint(
                root=tmp_path,
                captured_action_count=action_count,
                source_episode_id="episode_a",
                identity=identity,
                compatibility={"server": "v1"},
                simulator_state=np.arange(12, dtype=np.float32) + action_count,
                logical_state={"schema_version": "logical/v1", "value": action_count},
                carry_tracker_state={"schema_version": "carry/v1", "value": None},
                verifier={"leaf_truth": [True], "official_q": 0.5, "official_success": False},
                retained_action_counts=retained,
            )
        )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "checkpoint_action_0100.json",
        "checkpoint_action_0100.npy",
        "checkpoint_action_0150.json",
        "checkpoint_action_0150.npy",
    ]
    loaded = load_persistent_checkpoint(
        manifest_path=records[-1]["manifest_path"],
        expected_manifest_sha256=records[-1]["manifest_sha256"],
        expected_identity=identity,
        allowed_root=tmp_path,
    )
    assert loaded.captured_action_count == 150
    assert np.array_equal(
        loaded.simulator_state,
        np.arange(12, dtype=np.float32) + 150,
    )


def test_manifest_or_state_tampering_fails_closed(tmp_path) -> None:
    identity = {"task_id": 8}
    record = write_persistent_checkpoint(
        root=tmp_path,
        captured_action_count=50,
        source_episode_id="episode",
        identity=identity,
        compatibility={},
        simulator_state=np.arange(4, dtype=np.float32),
        logical_state={},
        carry_tracker_state={},
        verifier={},
        retained_action_counts=[50],
    )
    manifest_path = tmp_path / "checkpoint_action_0050.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_episode_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PersistentCheckpointError, match="manifest hash changed"):
        load_persistent_checkpoint(
            manifest_path=manifest_path,
            expected_manifest_sha256=record["manifest_sha256"],
            expected_identity=identity,
            allowed_root=tmp_path,
        )

    manifest_path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    new_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    state_path = tmp_path / "checkpoint_action_0050.npy"
    state_path.write_bytes(state_path.read_bytes() + b"x")
    with pytest.raises(PersistentCheckpointError, match="state hash changed"):
        load_persistent_checkpoint(
            manifest_path=manifest_path,
            expected_manifest_sha256=new_hash,
            expected_identity=identity,
            allowed_root=tmp_path,
        )


def test_identity_mismatch_fails_before_state_restore(tmp_path) -> None:
    record = write_persistent_checkpoint(
        root=tmp_path,
        captured_action_count=50,
        source_episode_id="episode",
        identity={"task_id": 0},
        compatibility={},
        simulator_state=np.arange(4, dtype=np.float32),
        logical_state={},
        carry_tracker_state={},
        verifier={},
        retained_action_counts=[50],
    )
    with pytest.raises(PersistentCheckpointError, match="identity changed"):
        load_persistent_checkpoint(
            manifest_path=record["manifest_path"],
            expected_manifest_sha256=record["manifest_sha256"],
            expected_identity={"task_id": 1},
            allowed_root=tmp_path,
        )
