# Independent integration review

A separate read-only review traced public task selection through resource
relocation, the original scene loader, model execution and result inspection.
It identified integration defects that were fixed before packaging:

| Finding | Correction and CPU check |
| --- | --- |
| CLI reserved output conflicted with the model wrapper | Explicit reservation accepts only the new private input directory; existing episodes remain rejected |
| Scene root and registry root were confused | Separate roots; exact ready/socket/PID ownership check before connecting |
| Source instance/name differed from the scene's public binding | Explicit boundary conversion and public instruction identity; source identity retained privately |
| Missing single-load YAML | Frozen resource imported and original loader tested |
| Resolving the Python symlink bypassed a venv | Preserve the invocation path; resolve only for identity comparisons |
| Startup failures and unavailable conversations disappeared from summaries | Preserve initialization records and report incomplete/invalid runs separately |
| Extra gates could upgrade a false original success flag | Require the original success flag as well as operational and declared extra gates |
| Process scoring could not be called from the CLI | `score --process-output` forwards a fresh audit directory |
| An exited launcher could leave child processes alive | Clean the owned process group even after leader exit; CPU subprocess regression |
| Live transport dependencies were implicit | Explicit `live` installation extra |

The review also identified the declared wall-clock budget mismatch. Investigation
confirmed it is present in the source implementation, so it is documented rather
than silently changed during packaging. Configuration symlink handling and
interrupted startup recording were checked in the coordinator's final integration.

The review ran no simulator, GPU, model request, SSH or resident connection.
CPU integration and wheel acceptance results are in [VALIDATION.md](VALIDATION.md).
This review is not a Live qualification or an approval of other task-family
execution profiles.
