# Model integration

The candidate executes the frozen Bench model loop from
`5b8d2b37fd7d97b161e88de4d438ec789c78f9b4`. `esi_benchmark.models` configures
that implementation; it does not introduce another episode controller.
No model SDK is required: the inherited transports use Python's HTTP stack.

The public config's `model` object contains:

| Field | Meaning |
| --- | --- |
| `endpoint` | Base API URL, e.g. `https://provider.example/v1`. No embedded credentials, query or fragment. |
| `model` | Provider model identifier. |
| `api` | `chat_completions` (default), `responses`, or `anthropic_messages`. |
| `api_key_env` | Explicit environment variable name, never the key itself. Optional for a local anonymous endpoint. |
| `timeout` | Integer HTTP timeout, 1–900 seconds; default 300. |
| `maximum_output_tokens` | Positive integer or `null`; omitted/null preserves the upstream unlimited-request behavior. This does not remove provider limits. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `preserve_thinking` | Optional generation settings, validated by the frozen provider. Defaults are recorded in the client manifest. |
| `reasoning_effort`, `reasoning_context`, `reasoning_mode`, `reasoning_summary`, `text_verbosity` | Adapter-specific settings below. |

For a local server, set `endpoint` to `http://127.0.0.1:8000/v1`, `model` to
its served model name, and omit `api_key_env` if it needs no key. Plain HTTP
retains the upstream restriction to loopback or private IPv4. For a hosted
server, use HTTPS and an explicitly named environment variable. The candidate
never falls back to an unrelated ambient provider key. Validation is offline;
provider capability preflight makes real model requests only during a Live run.

```python
from esi_benchmark.models import validate_model_config

validate_model_config({
    "endpoint": "https://provider.example/v1",
    "model": "your-model",
    "api": "chat_completions",
    "api_key_env": "ESI_MODEL_API_KEY",
    "timeout": 300,
})
```

The runner builds the exact public prompt, tools, images and history through
`EpisodeConversation`, calls the existing provider, and parses exactly one
native `robot_action` per assistant turn. The existing capability preflight,
invalid-output repair, provider retry accounting and terminal classifications
remain in force. Generation settings and request/response artifacts are stored
per run. Selecting a different adapter or history length is an explicit
experimental choice and must be reported. The frozen L0 sidecar uses 5 history
frames (`frozen_history_5`); 20 is an explicit variant
(`history_20_variant_not_baseline`), not the frozen baseline.

Responses requires explicit `reasoning_effort`, `reasoning_context`,
`reasoning_mode`, and `text_verbosity`; `reasoning_summary` is optional.
For example the tested CPU fixture uses `medium`, `all_turns`, `standard`, `low`.
The provider validates supported values against the imported adapter. Anthropic
Messages requires an explicit `reasoning_effort` and rejects Responses-only
settings. Both adapters retain the original native-tool conversion and opaque
reasoning replay. Their CPU contract tests pass; no provider endpoint was
contacted during candidate preparation. Hosted gateways differ, so each new
provider must pass capability preflight and the separate Live checklist.

`models.run_episode(...)` accepts an already owned independent scene's
`ready.json`, its runtime registry root, a fresh output directory, a frozen task
binding, and validated model config. It never discovers or attaches to unrelated
residents. The CLI owns scene startup/cleanup. The task binding includes the
numeric public handle, source task identity/seed/instruction and the exact
per-task action/model-call budgets. It is obtained from the prepared task bundle;
there is no universal 150-decision override. The implementation maximum remains
150, while individual contracts may be smaller.

The public runner does not expose the internal SSH scheduler, key-file defaults,
image upload proxy, gateway relay management, or automatic historical run resume.
Their historical helper code, where required by the frozen module closure, is
not a supported public command. Ordinary Chat Completions, Responses and
Anthropic requests use the explicitly configured endpoint directly.

Some frozen handoffs declare `wall_clock_seconds: 3600`, but the source Bench
loop has no consumer of the launcher-provided wall-clock environment variable.
The candidate records this declaration as `declared_budget` with
`wall_clock_enforcement: not_implemented_in_frozen_bench`; it does not silently
add a new terminal condition. Action-decision and model-call budgets are enforced.
Strict reproduction of a declared overall wall-clock deadline is therefore not
claimed. Scene-load deadlines and provider request timeouts remain separate.
