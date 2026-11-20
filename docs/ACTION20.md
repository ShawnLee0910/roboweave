# Action20 protocol

The authoritative resources are `action20_tool_schema.json`,
`action20_bare_marker_schema.json`, `action20_core_instructions.txt`, and
`action20_protocol.py` in the **selected profile's Bench root**. Resolve it with
`roboweave.profiles.source_roots(profile)["bench_root"]`; the default L0 root
is `_vendor/bench`, while other revisions live under `_vendor/revisions/bench/`.
All are installed package resources, independently frozen from their Runtime
bindings. See [Support matrix](SUPPORT_MATRIX.md) and
[import provenance](../provenance/imports_profiles.json).

The six Bench bundles share the same tool schemas and core instruction bytes.
Their implementation constants need not be identical: L2's parser has a
3,000-decision implementation ceiling, while the other frozen parsers use 150.
The task's declared budget is still the episode limit; neither ceiling replaces
it. A newer checkout HEAD does not redefine any of these profiles.

Action20 contains 19 robot actions plus `done`:

| Group | Actions |
| --- | --- |
| Base translation | `move_forward`, `move_backward`, `move_left`, `move_right` |
| Base rotation | `turn_left`, `turn_right` |
| Observation / idle | `look_up`, `look_down`, `wait` |
| Manipulation | `pick`, `open`, `close`, `toggle_on`, `toggle_off` |
| Placement | `place_on_floor`, `place_inside`, `place_on_top`, `place_under`, `place_next_to` |
| Termination | `done` |

Invoke one native function tool named `robot_action`, with one JSON argument
object and no ordinary assistant content. There is no batching or parallel
robot action. `target` is an integer referring to a label in the current Marker
observation; it is not an object ID. `target_description` records a brief
model-authored visual description. Backend object identities, scene predicates,
private verifier details, and scores are never model observations.

```json
{"action":"pick"}
```

This requests fresh markers. If the tool returns `selection_required`, select
using the same action:

```json
{"action":"pick","target":3,"target_description":"red cup"}
```

`pick`, `open`, `close`, `toggle_on`, `toggle_off`, and `place_under` use two
calls: request, then select. `place_inside`, `place_on_top`, and `place_next_to`
usually use three: request, select the destination, then select the displayed region when requested.
The current candidate labels and stage determine valid selections. Markers are
transient and cannot be reused from an older image. Changing the action without
a target cancels a pending selection and starts a new decision.

Navigation is relative to the robot base, with translation choices
`move_distance_m` in `{0.15, 0.30, 0.60}` and rotation choices `turn_angle_deg`
in `{10, 30}`. Defaults are 0.15 m and 10 degrees. Optional navigation `repeat`
is an integer 1–10; the runtime expands larger moves/turns into its existing
safe substeps. Arbitrary world coordinates and continuous arm commands are
not part of this interface. Exact preconditions and endpoint/collision checks
are retained in the frozen runtime, including failure and rollback behavior.

A Marker chain through final execution counts as one action decision. Switching
away from a pending chain consumes that decision and the replacement decision.
A repeated navigation counts as one decision. Provider/protocol retries consume
the internal model-call cap, not extra public action decisions. `done` does not
count as a robot action. At exhausted decision budget, only a pending Marker
chain may finish, after which the episode ends. The task's frozen budget may be
smaller than its profile's implementation ceiling (150 outside L2, 3,000 for L2).

Public feedback includes `completed`, `selection_required`, `blocked`, and
`execution_unverified`. A completed action is not proof of task success.
Execution-unverified means the scene may already have changed. Blocked feedback
uses coarse public categories; private exceptions and exact hidden state remain
private. Consecutive blocked and provider/protocol retry limits, recovery gates,
transport-timeout handling, and official terminal notifications retain their
original distinctions in `summary.json`.

Only the left hand manipulates objects. Empty-hand interactions and held-object
placement requirements remain enforced; visible markers do not guarantee reach
or collision-free placement. The frozen Symbolic OnTop path retains its existing
one-time support approach. Physical profiles retain their own native admission
and execution rules; the public action name does not authorize a new approach
or recovery policy.

The public profiles provide a head/left-wrist/right-wrist RGB composite (head on the
right; left wrist above right wrist on the left). The default
history configuration is the frozen sidecar's 5 completed physical frames; 20
is an explicit supported variant and must not be reported as the frozen baseline. A transient current observation may add one view. Marker
micro-steps remain in the trajectory while completed physical history follows
the original compression/selection rules. Optional RGB-D, overhead views,
automatic approach and other experimental execution modes are not silently
enabled by selecting a different family. Symbolic and Physical execution retain
their frozen backends, prerequisites and success gates; identical public action
names do not make their execution semantics interchangeable.

The WIPE experiment adds one two-call Marker action to `l0-english-v2`.
Set `ROBOWEAVE_EXPERIMENTAL_WIPE=1` for the run to enable it. Pick a
compatible cleaning tool, request a `wipe` Marker, then select a visible
target. Completion requires at least one original covering particle system
to clear and the same cleaning tool to remain held.

Run the synthetic replay for a concrete three-call Marker example. It exercises
the real parser, conversation state machine and writers, with synthetic scene
replies; it is not evidence of collision checks, manipulation or simulator
success.
