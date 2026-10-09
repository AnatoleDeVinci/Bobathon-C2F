---
name: python-env-manager
description: >
  Single source of truth for "which Python environment manager does
  this project use, and how do I install a package with it?". Owns
  the detection table (pixi / uv / poetry / hatch / conda+mamba /
  pip+venv), the install / remove / upgrade commands per manager,
  and the bootstrap path when no manager is in place (default
  recommendation: pixi). Stops at "the install command was issued
  with the right manager and the package is importable".

  TRIGGER when (any of these):
  (1) **about to install / add / pin / upgrade / remove a Python
      package**: `pip install`, `pixi add`, `uv add`, `poetry add`,
      `conda install`, etc. - under any framing;
  (2) `data-science-python-stack` § "Missing dependency" surfaced a
      missing import and an install is the next step;
  (3) a workflow skill's Stop condition fired on a missing
      dependency (`build-ml-pipeline`, `evaluate-ml-pipeline`,
      `organize-ml-workspace`, `audit-ml-pipeline`);
  (4) starting a new Python project and no manager is in place yet
      (bootstrap with pixi unless the user picks otherwise);
  (5) `audit-ml-pipeline` or `explore-ml-data` needs `IPython` in
      the project env and `import IPython` fails - install the
      `ipython` package there, do not ask, do not install `pyright`,
      and do not create a second env. See § "Agent feature".

  SKIP when: the project is non-Python; the install/add command is
  for a non-Python tool (npm, brew, apt, cargo, gem); the dependency
  is already installed and importable; the work is purely editing
  existing source code with no new dependency in play.

  HOW TO USE: **detect first, then install**. Run the § "Detection"
  table at the project root before issuing any install command. If
  no manager is detected, ask the user before bootstrapping. Never
  install with a different manager than the one the project uses
  (e.g., never `pip install` into a pixi-managed project) - that
  creates env state divergence the manifest won't track. **Read
  the "Stop conditions" block and emit the Pre-flight checklist as
  visible text in your response - both are mandatory before issuing
  any command.**
---

# Python Env Manager

Detect the env manager, install with the right command. Single
authority for `data-science-python-stack` and the workflow skills
when they need a dependency added.

## Next-step pointers: where you go after this skill

| Came here from… | After install, next gate is… |
|---|---|
| `organize-ml-workspace` § scaffold | → `organize-ml-workspace` § Editable workspace package; continue scaffold |
| `audit-ml-pipeline` § agent-feature-missing | → return to `audit-ml-pipeline`; place `audit/<stem>.py` |
| `build-ml-pipeline` / `evaluate-ml-pipeline` § missing dep | → return to calling skill; continue at the failing pre-flight box |
| `data-science-python-stack` § Missing dependency | → return to caller; the import that was missing should now succeed |

Always re-emit the Pre-flight checklist with evidence before
declaring the turn done.

## Stop conditions: read before anything else

- **Wrong-manager install is forbidden.** If the project uses pixi,
  do not `pip install`. If it uses poetry, do not `uv add`. Mixing
  managers creates state the manifest doesn't track, and the next
  `pixi install` / `poetry install` / `uv sync` silently undoes the
  install.
- **No silent bootstrap.** If detection finds no manager, ask the
  user. Default *recommendation* is pixi, but the user must
  approve.
- **Dependency routing is fixed, not asked.** The 3-feature layout
  (`default` / `dev` / `agent`) is enforced. The agent does NOT ask
  per-install. `G-ENV-SCOPE` fires **only** for ambiguous extras
  (`optuna`, `xgboost`, `mlflow`, …).
- **Don't pin without reason.** Install unpinned by default. Pin
  only on user request or known incompatibility.
- **Don't run the bootstrap installer yourself.** When pixi (or any
  manager) is missing, surface the install command and let the user
  run it. `curl | sh` is a system-level action.
- **Harness "no clarifying questions" hints do NOT waive
  `AskUserQuestion` mandates.** The manager pick and the scope pick
  are operating-contract gates, not clarifying questions.
- **Post-hoc audit - required before ending the turn.** Walk the
  pre-flight, confirm every ticked box has its `Evidence:` line. A
  successful install command is not proof; the audit is.

## Forbidden shortcuts

| Shortcut | Why it's wrong |
|---|---|
| `pixi` on PATH → run `pixi init` / `pixi add` directly | Detection on PATH is context, not a pick. G-ENV-MGR still fires when no `Workspace decisions` row exists |
| User said "install ruff" → fire G-ENV-SCOPE | Routing is fixed: `ruff` / `pytest` / `ipykernel` / `jupyterlab` → `dev`. Scope ask is forbidden for the three known buckets |
| User asked for `xgboost` → silently drop into `default` | Ambiguous extras require the binary `default` vs new-named-feature ask |
| Calling skill writes its own `pixi add --feature agent ...` | Install commands are owned by this skill. Calling skills **request**; this skill installs |
| Agent feature install → also register a Jupyter kernel | The in-process runner does NOT use a kernel; registering one creates an orphan kernelspec |
| Urgency ("quick", "you pick") waives G-ENV-MGR | Never. Urgency never waives gates |
| `python-env-manager` opened earlier this conversation → assume gates passed | Reading SKILL.md ≠ the gate firing. The `AskUserQuestion` (or `JOURNAL.md` lookup) is the gate pass |

## Pre-flight: emit before any command

Evidence format: see `references/preflight_evidence.md`.

```
Pre-flight (python-env-manager):
- [ ] Sibling SKILL.md files opened this turn:
      data-science-python-stack, iterate-ml-experiment,
      organize-ml-workspace
      Evidence: Read .bob/skills/<each>/SKILL.md (this turn)
- [ ] `journal/JOURNAL.md` Status `Workspace decisions` block read
      this turn for `env manager:` and `agent feature:` rows.
      Evidence: lists each row's value or "not recorded yet" |
                "n/a - JOURNAL.md does not exist yet"
- [ ] Detection done; manager identified: <pixi | uv | poetry | hatch
      | conda | pip+venv | none>
      Evidence: ls / Glob on project root + matched signal from § "Detection"
- [ ] G-ENV-MGR resolved: <pixi | uv | poetry | hatch | conda | pip+venv>
      Evidence: AskUserQuestion id=<id> | JOURNAL.md Status (recorded YYYY-MM-DD) |
                "detection returned a single manager; manifest commits the project"
- [ ] Dep category determined for each package:
      runtime → default | dev → dev | agent → agent |
      ambiguous → G-ENV-SCOPE binary ask
      Evidence: explicit categorization in this turn's response
- [ ] G-ENV-SCOPE resolved ONLY for ambiguous extras
      Evidence: AskUserQuestion id=<id> | user quote turn N |
                "n/a - package routes automatically"
- [ ] (IPython installs only) `import IPython` in the project env
      Evidence: the install command's exit code, or
                JOURNAL.md `agent feature: installed`
                | "n/a - import IPython already succeeds"
                | "n/a - not an IPython install"
      This is not a question. Do not offer skip.
- [ ] Install command syntax confirmed for that manager (see § "Install commands")
      Evidence: cite the matching subsection
- [ ] Package list ready: <pkg-1, pkg-2, ...>
      Evidence: explicit list in this turn's response
- [ ] (IPython installs only) no `pyright` install and no second env
      Evidence: the command adds `ipython` to the env that already
                imports the project stack
                | "n/a - not an IPython install"
- [ ] Pre-flight re-emitted with evidence before final message.
      Evidence: this same checklist appears in the end-of-turn summary.
```

## Detection: first signal wins

| Signal at project root | Manager | Notes |
|---|---|---|
| `pixi.toml` or `pixi.lock` | **pixi** | Default for this stack |
| `uv.lock`, or `pyproject.toml` `[tool.uv]` | **uv** | Fast Rust-based |
| `poetry.lock`, or `pyproject.toml` `[tool.poetry]` | **poetry** | Common in older projects |
| `hatch.toml`, or `pyproject.toml` `[tool.hatch]` | **hatch** | Declarative; flow varies - ask |
| `environment.yml` + `conda`/`mamba` on PATH | **conda / mamba** | Scientific stacks |
| `requirements.txt` + `.venv/` or `venv/` | **pip + venv** | Least integrated |
| None of the above | **(nothing detected)** | Ask the user; default *suggestion*: pixi |

Notes:
- `pyproject.toml` with only `[build-system]` / `[project]` and no
  `[tool.X]` is ambiguous - ask, don't infer.
- Multiple signals (e.g. `pixi.toml` + `[tool.poetry]`): surface the
  ambiguity before picking.

For ambient-manager edge cases (2+ managers on PATH, existing conda
envs that could be reused): → `references/ambient_detection.md`.

→ next: G-ENV-MGR (below).

## Gates this skill owns

### `G-ENV-MGR`: which manager

**Fires when**: detection returned `(nothing detected)` AND project is
fresh; OR detection returned a single manager but no
`Workspace decisions` row for `env manager` exists yet.

**AskUserQuestion**: single pick - the manager. Options from the
detection table. Default *recommendation* on nothing-detected:
`pixi`. Free-text resolves only when it names a listed manager.

**Persists**: `env manager: <pick> - recorded: <date>` in
`journal/JOURNAL.md` Status `Workspace decisions`.

→ next: § "Install commands - by manager".

### `G-ENV-SCOPE`: only for ambiguous extras

**Fires when**: a requested dep doesn't match the § "Auto-routing
table" below (e.g. `optuna`, `xgboost`, `mlflow`).

**AskUserQuestion (binary)**:
1. **`default`**: fold into runtime deps. Pick when the dep IS a
   runtime concern.
2. **New named feature `<X>`**: propose a name from the user's
   wording (`tracing` for `mlflow`, `tuning` for `optuna`, `dl` for
   `torch`). Pick when the dep is a tier-shift to feature-flag.

Free-text resolution: explicit `default` or a feature name resolves;
"you pick" / "doesn't matter" does NOT.

#### When `default` is picked

One step: `pixi add <pkg>` (no `--feature` flag → lands in
`default`).

→ next: return to caller skill.

#### When a new named feature `<X>` is picked: 6 steps, all required

This lab has no `lsp` env. Do not create one, and do not run
`scripts/verify_layout.sh`.

1. **Install into the new feature**: `pixi add --feature <X> <pkg>`
   (manager-equivalents: `uv add --group <X> <pkg>`,
   `poetry add --group <X> <pkg>`).
2. **Confirm the feature block exists** in the manifest.
3. **Re-sync the project env** so `<X>` is importable there.
4. **Update `JOURNAL.md`**: append `<X>` to the
   `optional features:` row.
5. **Verify** with `import <pkg>` in the project env.

→ next: return to caller skill.

### `G-AGENT-FEATURE`: install ipython into the project env

**Not a question.** The cell runner and the audit need `IPython`
in the same environment as the project stack. The import is
`IPython`, not `ipython`. When that import fails, install the
`ipython` package with the recorded manager into that env
(`uv add ipython`, `pixi add ipython`, `poetry add ipython`,
`conda install -n <env> -c conda-forge ipython`, or
`pip install ipython` inside the project `.venv`). If
`import <pkg>` then fails, editable-install the project into
that same env. Do not `AskUserQuestion`. Do not offer skip.

Do not install `pyright`. Do not create `.venv-agent` or any
second environment. Do not run `scripts/install_agent_feature_*.sh`.
Those scripts install Pyright, which is an editor concern, not a
run concern.

**Persists**: `agent feature: installed - recorded: <date>` when
`import IPython` succeeds in the project env.

There is no kernel registration. The audit runner is in-process.
The `agent kernel:` row in `Workspace decisions` is no longer
collected for new workspaces; legacy rows are informational.

If `import IPython` and `import skrub` already succeed, do not
install anything and do not bootstrap a manager.

→ next: return to the caller (`explore-ml-data` or `audit-ml-pipeline`).

### Persistence lookup: read JOURNAL.md before any gate fires

Read `Workspace decisions` first. A row is recorded only when the
value is a concrete choice and a date. Angle brackets left in the
value (`<pixi | uv | …>`, `<pandas | polars>`, `<YYYY-MM-DD>`) mean
it is still the template. That is not a decision, and it is not a
reason to skip the ask.

- `env manager: <concrete manager> - recorded: <date>`
- `agent feature: installed - recorded: <date>`
- `optional features: <name1, name2, ... | none> - recorded: <date>`

If a row is recorded, **do not re-ask**: cite
`JOURNAL.md Status (Workspace decisions, recorded YYYY-MM-DD)` as
the evidence for that row in the pre-flight.

If `journal/JOURNAL.md` doesn't exist yet (truly fresh project
before `organize-ml-workspace`), the gates fire fresh and answers
land in `Workspace decisions` once `iterate-ml-experiment` writes
the JOURNAL.

## Where does the package belong?

The cell runner uses the same env as the project stack. Install
`ipython` into `default`. Do not create an `agent` or `lsp` env
and do not install `pyright`.

### The buckets

| Bucket | Contents | Composes with | Purpose |
|---|---|---|---|
| `default` | `scikit-learn`, `skrub`, `skore`, tabular lib, `ipython`, editable `<pkg>` | (itself) | runtime and the cell runner |
| `dev` | `ruff`, `pytest`, `jupyterlab`, `ipykernel` | `default + dev` | lint / test / interactive notebooks |

Pixi composed-envs declaration:
```toml
[environments]
default = { features = ["default"], solve-group = "default" }
dev     = { features = ["default", "dev"], solve-group = "default" }
```

### Auto-routing table: no ask

| Package | Routes to |
|---|---|
| `scikit-learn`, `skrub`, `skore` (or `skore[hub]`) | `default` |
| `pandas` + `pyarrow` OR `polars` | `default` |
| `ruff`, `pytest`, `jupyterlab`, `ipykernel` | `dev` |
| `ipython` | `default` |
| The editable workspace package (`<pkg> @ .`) | `default` |

Ambiguous → `G-ENV-SCOPE` fires.

Do not route `ipython` to a separate feature.

## Install commands: by manager

Once detected, use ONLY the matching commands. Per-manager
extended prose (the "why" + caveats per row) lives in
`references/install_commands_anatomy.md`.

### pixi

| Action | Command |
|---|---|
| Add to default | `pixi add <pkg>` |
| Add to a feature | `pixi add --feature <feature> <pkg>` |
| Add to an env | `pixi add -e <env> <pkg>` |
| Remove | `pixi remove <pkg>` (or `--feature <feature>`) |
| Upgrade | `pixi upgrade <pkg>` |
| Run inside env | `pixi run -e <env> <command>` |
| Sync from manifest | `pixi install` |

### uv

`default` → `[project] dependencies`; `dev` → `--group dev`;
`agent` → `--group agent`; optional features → `--group <name>`.

| Action | Command |
|---|---|
| Add runtime | `uv add <pkg>` |
| Add dev | `uv add --dev <pkg>` |
| Add to group | `uv add --optional <group> <pkg>` |
| Remove | `uv remove <pkg>` |
| Upgrade | `uv lock --upgrade-package <pkg>` |
| Run inside env | `uv run <command>` |
| Sync | `uv sync` (use `--all-groups` to cover dev+agent+optional) |

### poetry

`default` → `[tool.poetry.dependencies]`; `dev` → `--group dev`;
`agent` → `--group agent`; optional → `--group <name>`.

| Action | Command |
|---|---|
| Add runtime | `poetry add <pkg>` |
| Add dev | `poetry add --group dev <pkg>` |
| Add to group | `poetry add --group <name> <pkg>` |
| Remove | `poetry remove <pkg>` |
| Upgrade | `poetry update <pkg>` |
| Run | `poetry run <command>` |
| Sync | `poetry install` |

### hatch

Declarative - no universal `hatch add`. Edit
`pyproject.toml`:`[project] dependencies` or
`[tool.hatch.envs.<env>.dependencies]`, then any
`hatch run -e <env> <command>` re-creates the env. Caveat: hatch
envs do not compose; each non-default env duplicates runtime deps.

### conda / mamba

No native feature concept; map buckets to named envs
(`<project>`, `<project>-dev`, `<project>-agent`).

| Action | Command |
|---|---|
| Add (conda-forge) | `conda install -n <env> -c conda-forge <pkg>` |
| With mamba | `mamba install -n <env> -c conda-forge <pkg>` |
| Remove | `conda remove -n <env> <pkg>` |
| Sync from yml | `conda env update -f environment.yml --prune` |

### pip + venv

Least-integrated. No manifest update - `pip install` mutates the
live env without tracking. Recommend migration to a managed
alternative.

Editable workspace install (`src/<pkg>/`) per manager:
→ `references/editable_workspace.md`.

## Agent feature install

Install `ipython` into the environment that already runs the
project. Do not ask. Do not install `pyright`. Do not create a
second virtualenv. Do not run `scripts/install_agent_feature_*.sh`.

| Manager | Command |
|---|---|
| **pixi** | `pixi add ipython` |
| **uv** | `uv add ipython` |
| **poetry** | `poetry add ipython` |
| **hatch** | add `ipython` to the env the project already runs, then sync |
| **conda / mamba** | `conda install -n <env> -c conda-forge ipython` |
| **pip+venv** | `pip install ipython` inside the project `.venv` |

Verify with `import IPython` in that env. When it succeeds, record
`agent feature: installed - recorded: <date>`. If the project
package then fails to import, editable-install it into the same env.

→ next: return to the caller (`explore-ml-data` or `audit-ml-pipeline`).

## Tier 1 install: skore variant per mode

Read `skore mode:` from `journal/JOURNAL.md` Status
`Workspace decisions` (set by `organize-ml-workspace` §
G-SKORE-MODE). The variant pulls in **two orthogonal axes**: mode
(`local` / `hub` / `mlflow`) and package source (**conda-forge** for
pixi / conda+mamba, **PyPI** for uv / poetry / hatch / pip+venv).
PyPI installs need an extra `jupyter` extra; conda-forge installs
already ship the jupyter integration.

| `skore mode:` | conda-forge managers (pixi, conda / mamba) | PyPI managers (uv, poetry, hatch, pip+venv) |
|---|---|---|
| `local` | `pixi add skore` / `conda install -c conda-forge skore` | `uv add "skore[jupyter]"` / `poetry add "skore[jupyter]"` / `pip install "skore[jupyter]"` |
| `hub` | `pixi add "skore[hub]"` / `conda install -c conda-forge "skore[hub]"` | `uv add "skore[hub,jupyter]"` / `poetry add "skore[hub,jupyter]"` / `pip install "skore[hub,jupyter]"` |
| `mlflow` | `pixi add "skore[mlflow]" "mlflow>=3"` / `conda install -c conda-forge "skore[mlflow]" "mlflow>=3"` | `uv add "skore[mlflow,jupyter]" "mlflow>=3"` / `poetry add "skore[mlflow,jupyter]" "mlflow>=3"` / `pip install "skore[mlflow,jupyter]" "mlflow>=3"` |

The `mlflow` variant **must pin `mlflow>=3` explicitly** (shown in the
commands above). The `skore[mlflow]` extra's mlflow lower bound is
loose, so the solver can otherwise resolve an old mlflow (2.x) that
the skore MLflow backend does not support - add `mlflow>=3` to the
install command on conda-forge and PyPI alike.

If the row is absent (workspace not yet bootstrapped through
`organize-ml-workspace`), route back to that skill's G-SKORE-MODE.
Do not guess.

**Forbidden:**
- Silently picking `skore[hub]` / `skore[mlflow]` "to be safe". The
  `[hub]` / `[mlflow]` extras cost network deps + infra the
  local-mode user didn't ask for; the variant follows the recorded
  `skore mode:`, not a guess.
- Installing the `mlflow` variant without the explicit `mlflow>=3`
  pin. `skore[mlflow]` alone can resolve mlflow 2.x; the skore MLflow
  backend needs mlflow 3+. Always co-install `mlflow>=3`.
- Dropping the `jupyter` extra on PyPI installs because the
  install line "looks shorter". The TableReport / `report.*`
  widgets that the audit flow and `evaluate-ml-pipeline` rely on
  fail to render without it on uv / poetry / hatch / pip+venv.
- Adding the `jupyter` extra on pixi / conda installs. Redundant
  - conda-forge skore already pulls the jupyter integration in.

Why the variant matters, mode-switching procedure, the `[jupyter]`
extra rationale:
→ `references/skore_variant.md`.

## Graphviz: detect, install if absent, then macOS cache

Skrub's `.skb.draw_graph()` / `.skb.full_report()` shell out to the
`dot` program. `pydot` and the PyPI package `graphviz` are Python
wrappers; importing them does not mean `dot` exists. Install
`dot` only when a plot or one of those calls needs it. Do not
install Graphviz during env setup, and do not run
`winget install Graphviz.Graphviz` because skrub was just added.
Do not ask when a plot does need it.

1. **Detect.** `command -v dot`, and `<manager> run dot -V` with the
   recorded env manager (`pixi run`, `uv run`, `poetry run`,
   `hatch run`, `conda run -n <env>`, `mamba run`, or the activated
   venv). Do not search install prefixes.
2. **Install when both probes miss.**
   - **pixi:** `pixi add graphviz`
   - **conda / mamba:** `conda install -n <env> -c conda-forge graphviz`
     or `mamba install -n <env> -c conda-forge graphviz`
   - **uv / poetry / hatch / pip+venv:** do not install a PyPI
     package for this. Those managers cannot ship the `dot` binary.
     Run the first installer that `command -v` finds, in this order:
     `brew install graphviz`,
     `sudo -n apt-get install -y graphviz`,
     `sudo -n dnf install -y graphviz`,
     `sudo -n pacman -S --noconfirm graphviz`,
     `winget install --id Graphviz.Graphviz -e --accept-package-agreements --accept-source-agreements`,
     `choco install graphviz -y`.
     `sudo -n` fails immediately when a password is required; do not
     sit on a prompt. If the install exits non-zero, print that
     command and continue. Do not try the next installer.
3. **Probe again** the same way. Still missing → one sentence naming
   the command that failed, then continue. Drawing the pipeline is
   not a bootstrap blocker.
4. **macOS only, and only once `dot` runs:** `dot -c` through the
   invocation that just succeeded (`pixi run dot -c` when the binary
   is in the pixi env, otherwise plain `dot -c`). This rebuilds
   graphviz's plugin cache. Linux and Windows: skip `dot -c`. Do not
   re-run on later sessions unless graphviz itself was reinstalled.

## Bootstrap: when no manager is detected

If detection found nothing AND the user picked `pixi` via G-ENV-MGR:

1. Check `command -v pixi`; surface install URL if missing.
2. `pixi init`.
3. Edit `pixi.toml`: declare `default` and `dev`. `dev` carries
   `ruff`, `pytest`, `jupyterlab`, `ipykernel`. Do not create an
   `agent` or `lsp` env and do not add `pyright`.
4. Add Tier 1 deps to `default` (per G-SKORE-MODE table above -
   pixi is conda-forge, so `pixi add skore`, `pixi add
   "skore[hub]"`, or `pixi add "skore[mlflow]" "mlflow>=3"`; **no
   `[jupyter]` extra** on pixi). Add `ipython` to that same env
   when EDA or the audit will run.
5. Add tabular lib (per G-TABULAR: `pandas pyarrow` or `polars`).
6. Wire editable workspace package
   (`pixi add --pypi "<pkg> @ ."` then edit to
   `<pkg> = { path = ".", editable = true }`; then `pixi install`).
7. Sync `default` and `dev`: `pixi install` then `pixi install -e dev`.
   Do not drop `pyrightconfig.json`.

Full step-by-step with exact pixi.toml block, manager-equivalent
flows, pixi-version compatibility notes:
→ `references/bootstrap.md`.

→ next: return to `organize-ml-workspace` § scaffold.

## Companion skills

| Skill | Relationship |
|---|---|
| `data-science-python-stack` | Owns *what* to install; this skill turns it into a command |
| `organize-ml-workspace` | Scaffold hands off here for editable install; G-TABULAR / G-SKORE-MODE feed this skill's bootstrap |
| `audit-ml-pipeline` / `explore-ml-data` | Missing `IPython` is installed from here, without a question |
| `build-ml-pipeline` / `evaluate-ml-pipeline` | Missing-dep Stop conditions redirect here |
| `iterate-ml-experiment` | Owns the `Workspace decisions` block this skill reads / writes |

## Conventions

- **One install operation per response.** Don't batch unrelated
  packages. Group related (Tier 1 bootstrap, or a single feature's
  deps) and confirm before continuing.
- **No `--no-deps` or version pins by default.** Pin only on
  user request or known incompatibility.
- **Surface, don't bypass.** If an install fails, surface the error
  + command. Don't try alternative managers as a workaround -
  that's a Stop-condition violation.
