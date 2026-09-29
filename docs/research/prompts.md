# Starting future research tasks

These are optional user prompts, not extra startup instructions. Replace bracketed
fields with the scope of the task. The repository guidance supplies persistent
host, evidence, and preservation constraints.

## Recover a model

> Identify the checkpoint(s) behind [method, dataset/scene, protocol, table/row].
> Return host, exact model/iteration, split/config, metric source, selection rule,
> and evidence status. Check the owning filesystem and explain any unresolved
> fields. Finish with a usable model choice for [rendering/ablation/resumption]
> and its limitations. This is an inspection task; do not train.

## Revise a paper section

> Revise [file/section] to address [reviewer concern]. Use the existing audit and
> implementation evidence, preserve supported technical meaning, and prefer the
> minimum additional experiments. Complete the edits, compile, inspect affected
> claims and layout, and update the issue map. Identify any evidence gap that the
> requested revision cannot resolve honestly.

## Run a bounded ablation

> Test whether [mechanism] changes [outcome] on [scene, split] from [checkpoint].
> Compare [variants] with [selection rule], using [iteration/time budget]. Run on
> the project's permitted host/GPU, preserve all runs, and continue through
> evaluation. Return the commands, configs, model paths, full metrics, relevant
> failures, and what the comparison establishes.

## Design before running

> Design the smallest experiment that distinguishes [hypothesis A] from
> [hypothesis B]. Inspect reusable outputs first. Produce concrete variants,
> controls, commands, budget, selection rule, and interpretation criteria. Stop
> with the experiment plan; this request does not include launching it.
