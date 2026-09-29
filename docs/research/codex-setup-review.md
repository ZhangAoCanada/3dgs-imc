# Codex instruction review — 26 September 2026

Follow-up, 27 September: the user replaced the GPU restriction with a choice of
physical GPU 0 or 1 per run. [AGENTS.md](../../AGENTS.md) and
[experiments.md](experiments.md) carry the current rule; GPU assignments below
describe the original instruction review only.

The active GS-LSS instructions have been revised in the main repository and its
`hust2` companion. The change follows the requested
[OpenAI article](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra):
keep persistent instructions focused, route context by task, remove unnecessary
procedural scaffolding, and make completion and decision boundaries explicit.
The scientific constraints come from the user's research policy and the existing
checkpoint audit.

## Findings and changes

| Finding | Resolution |
| --- | --- |
| Local `AGENTS.md` was empty; `AGENTS.override.md` held 8,184 bytes of instructions. | Made `AGENTS.md` the canonical 3,293-byte entry point and archived/removed the override. |
| `hust2` had a 30,236-byte, 544-line override with accumulated July policies. | Installed the same concise root instructions and removed the archived override. |
| Required reading loaded histories, multiple plans, drafts, and code before substantial work. | Added task-specific routes; raw history and large inventories are consulted only for relevant questions. |
| Active-paper/review paths referred to files absent on both machines. | Documented the submitted reference, actual local editing draft, distinct remote draft, and recovered review sources. |
| Remote GPU 2/local-only rules conflicted with the current policy. | Encoded GS-LSS on main GPU 1; baselines on `hust2` GPU 3, with main GPU 1 fallback. |
| Instructions encouraged repeated tuning toward a predetermined claim and preserving unsupported table narratives. | Required explicit experiment questions, selection rules, evidence-backed claims, and retention of negative results. Missing provenance does not itself justify a numerical replacement. |
| Every long run required a detailed watchdog/sleeper routine and automatic continuation into old queues. | Kept process/log/completion traceability; monitoring now follows the requested run and its actual lifecycle. |
| Broad escalation language could repeat approval requests for already authorized work. | Clarified continued inspection, SSH, reversible edits, and scoped verification within the task. Platform permissions still apply. |
| A paper folder is a nested Git repository; the remote repository lacks usable Git metadata. | Added short paper-folder instruction entry points so direct sessions can reach the parent policy. |

The root instruction text is **59.8% smaller locally and 89.1% smaller on
`hust2`**, measured in UTF-8 bytes. These are file-size reductions, not measured
latency or token-use improvements. Detailed references are loaded on demand.

## Active files

- [AGENTS.md](../../AGENTS.md): research constraints, task routing, and completion.
- [state.md](state.md): manuscript/host identities, current implementation, and
  links into the frozen history and evidence audit.
- [checkpoints.md](checkpoints.md): model lookup, interpretation of evidence,
  selection rules, missing sources, and checkpoint limitations.
- [experiments.md](experiments.md): host/GPU/environment policy, dataset pitfalls,
  run records, monitoring, and timing comparability.
- [paper-revision.md](paper-revision.md): source review letter, revision choices,
  implementation/claim risks, issue tracking, and compilation.
- [prompts.md](prompts.md): four optional task prompts with concrete deliverables
  and stopping conditions; these are not additional startup instructions.

Local paper entry points are `writings/GS-LSS-editing/AGENTS.md` and
`writings/GS-LSS-submission/AGENTS.md`. The remote entry point is
`hust2:/data1/zhangao/repos/3dgs-imc-codex/writings/GS-LSS/AGENTS.md`.

A byte-identical [review-letter copy](references/TVCG-2026-04-0389.txt) and its
[source record](references/source.json) are available on both hosts. The letter
is the original rejection/resubmission decision, not an invented major-revision
request. The existing 54-file audit package was also copied to `hust2` so its
lookup command and context links work there. This copied metadata does not move
any checkpoint or change its recorded host; the original local audit is unchanged.

## Other instruction surfaces reviewed

| Surface | Finding and treatment |
| --- | --- |
| Repository `.codex/` and `.agents/skills/` on both hosts | No active custom project config or skills to migrate; no new generic research skill was added. |
| `.aris/installed-skills-codex.txt.prev` | Historical inventory of 79 skills, with no active project links. Preserved as history; not reinstalled. |
| Local user config | Already selects `gpt-6-astra` with `max` reasoning and trusts this repository. Unchanged. |
| Remote user config | Selects `gpt-6-sol` with `xhigh` reasoning. Unchanged; model choice was not silently migrated across the user's other work. |
| Remote `~/.codex/AGENTS.md` | Reviewed the 6,608-byte shared research profile. Preserved its compatible user preferences; repository-specific instructions now supply the current paths and constraints. |
| System and installed plugin skills | Shared managed capabilities were not rewritten as project prompts. The task used the OpenAI Docs skill; no bulk skill activation was introduced. |
| Older remote workspace | No active instruction/skill files found in `/data1/ZHANGAo_2024/3dgs-imc-codex`. Preserved as a historical source. |
| Raw chats, Claude memory, research scripts, and manuscript text | Preserved. They are evidence or implementation, rather than a place to retroactively rewrite instructions. |

## Verification and limits

Checked relative document links on both machines, instruction discovery paths,
the identical shared root policy, backup hashes, and the transferred audit files.
Ran the documented checkpoint lookup on both hosts for a linked Table I source
and the unresolved GS-LSS Table III candidates. Both remote lookups correctly
retain `hust10` as the checkpoint host. The latter candidates remain explicitly
marked as not reproducing the submitted aggregates.

All 68 protected local files (including the original audit, selected code,
manuscripts, and user config) retained their hashes. All 15 fingerprinted remote
research sources also matched the original audit. Pre-existing tracked worktree
status and the remote shared instruction profile were unchanged. Detailed check
results are in [codex-setup-checks.json](codex-setup-checks.json).

A manual instruction review covered a small wording edit, checkpoint lookup,
missing Table III evidence, an experiment-plan-only request, a baseline run,
and a direct paper-folder session. These checks examine routing and constraints;
they are not a live-model A/B benchmark. No training, rendering, package
installation, scientific-code change, or paper-result edit was performed. Future
scientific performance still depends on the task, model, and actual experiments.

## Reload and recovery

Codex gives `AGENTS.override.md` precedence over `AGENTS.md` in a directory and
loads its instruction chain at session startup. Removing the stale overrides
makes the canonical file discoverable. Start a fresh session in the appropriate
repository for a clean load; this existing chat can still contain earlier
instructions in its history. See the
[official instruction discovery documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

The [backup archive](codex-instructions-before-2026-09-26.tar.gz) and
[hash manifest](codex-instructions-before-2026-09-26.json) are present on both
hosts. Members `local/AGENTS.md`, `local/AGENTS.override.md`, and
`hust2/AGENTS.override.md` preserve the replaced files. `hust2-user/AGENTS.md`
preserves the reviewed shared profile, which was not modified. Restore only the
intended host's member if rollback is desired; restoring an override makes it
active again. Do not unpack this historical archive over the workspace casually.
