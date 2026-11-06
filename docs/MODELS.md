# Model integration

The candidate executes the common model loop from the task's exact frozen Bench
bundle. The eight execution profiles select six Bench revisions; the default
remains `l0-english-v2`. [Support matrix](SUPPORT_MATRIX.md) records every binding.
This common harness does not claim to reproduce historical native model runners,
prompts or scheduler campaigns. The historical CODEX/AgentCLI harness is not a
supported model entry point. L0/L1 support the three HTTP adapters below;
the frozen L2 Bench supports only Chat Completions and Responses.

`roboweave.models.validate_model_config(..., profile=...)` and
`run_episode(..., profile=...)` use an isolated child process. Importing or calling
the public API does not inject legacy paths or modules into the caller. The
`create_provider` helper is private to the worker and is not a public API.
The inherited transports use Python's HTTP stack; no model SDK is required.

The public config's `model` object contains:

| Field | Meaning |
| --- | --- |
| `endpoint` | Base API URL, e.g. `https://provider.example/v1`. No embedded credentials, query or fragment. |
| `model` | Provider model identifier. |
| `api` | `chat_completions` (default), `responses`, or `anthropic_messages` (L0/L1 only). |
| `api_key_env` | Explicit environment variable name, never the key itself. Optional for a local anonymous endpoint. |
| `timeout` | Integer HTTP timeout, 1–900 seconds; default 300. |
| `maximum_output_tokens` | Positive integer or `null`; omitted/null preserves the upstream unlimited-request behavior. This does not remove provider limits. |
| `temperature`, `top_p`, `top_k`, `repetition_penalty`, `preserve_thinking` | Optional generation settings, validated by the frozen provider. Defaults are recorded in the client manifest. |
| `reasoning_effort`, `reasoning_context`, `reasoning_mode`, `reasoning_summary`, `text_verbosity` | Adapter-specific settings below. |

For a local server, set `endpoint` to `http://127.0.0.1:8000/v1`, `model` to
its served model name, and omit `api_key_env` if it needs no key. Plain HTTP
retains the upstream restriction to loopback or private IPv4. For a hosted
server, use HTTPS and an explicitly named environment variable. The candidate
never falls back to an unrelated ambient provider key. Configuration validation
checks the variable name without reading its value. An actual CLI Live run
requires that variable to be nonempty before resolving assets or loading a scene;
`--dry-run` needs no credential value. Provider capability preflight makes real
model requests only during a Live run.

The candidate disables automatic HTTP redirects for all three adapters,
including capability preflight. A 301, 302, 303, 307 or 308 response is a
non-retryable `provider_http_<status>` failure; no credentials or request body
are forwarded to the redirect target, even on the same origin. Configure the
provider's final API base URL directly. This candidate transport safeguard
does not change successful request payloads or the frozen evaluation protocol.
Provider diagnostics redact the credential actually sent, including standard
JSON escape forms, before truncating or saving a preview. Captured response-body
length and SHA256 remain unchanged. Unsafe or unparseable diagnostic encodings
use a fixed preview so logging cannot expose a partial key or fail on Unicode.

```python
from roboweave.models import validate_model_config

validate_model_config({
    "endpoint": "https://provider.example/v1",
    "model": "your-model",
    "api": "chat_completions",
    "api_key_env": "ROBOWEAVE_MODEL_API_KEY",
    "timeout": 300,
}, profile="l1-m30-public-v2")
```

The runner builds the exact public prompt, tools, images and history through
`EpisodeConversation`, calls the existing provider, and parses exactly one
native `robot_action` per assistant turn. The existing capability preflight,
invalid-output repair, provider retry accounting and terminal classifications
remain in force. Generation settings and request/response artifacts are stored
per run. Selecting a different adapter or history length is an explicit
experimental choice and must be reported. The frozen L0 sidecar uses 5 history
frames (`frozen_history_5`). The public profiles retain five as their default;
20 is an explicit variant (`history_20_variant_not_baseline`), not the frozen
baseline. Micro-step observations are distinct from completed physical history.

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
there is no universal 150-decision override. The non-L2 Bench implementation
ceiling is 150; L2's frozen Bench ceiling is 3,000. Each task still uses its own
smaller decision and model-call budgets; 3,000 is not a per-task allocation.
The CLI checks the seed against the native prepared inputs. Direct Python
callers must supply that verified seed: the inherited scene status protocol
reports task and instance identity, but does not attest the scene's seed.

The public runner does not expose the internal SSH scheduler, key-file defaults,
image upload proxy, gateway relay management, or automatic historical run resume.
Their historical helper code, where required by the frozen module closure, is
not a supported public command. Ordinary Chat Completions, Responses and
Anthropic requests use the explicitly configured endpoint directly.

L0/L1 handoffs declare `wall_clock_seconds: 3600`, but those frozen common Bench
loops do not consume the declaration. Their run records use
`wall_clock_enforcement: not_implemented_in_frozen_bench`; action-decision and
model-call budgets remain enforced. L2 collection has a distinct 7,200-second
contract and documented outer watchdog. See [Runtime](RUNTIME.md) for its timing,
safe-point checks and incomplete-result classification. Scene-load deadlines and
individual provider request timeouts are separate from either policy.
