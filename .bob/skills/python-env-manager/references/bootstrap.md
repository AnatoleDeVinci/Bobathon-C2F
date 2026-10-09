# Python Env Manager: Bootstrap detail

Full step-by-step pixi bootstrap when no manager is detected.
Cross-referenced from SKILL.md § Bootstrap. Load when scaffolding
a fresh workspace.

## Pre-conditions

- Detection (SKILL.md § Detection) found nothing at the project
  root.
- G-ENV-MGR fired and the user picked `pixi` (or another manager -
  mirror the same flow with the per-manager equivalents from
  SKILL.md § "Install commands - by manager").
- G-SKORE-MODE has resolved (SKILL.md § "Tier 1 install: skore
  variant per mode" reads the `skore mode:` row).
- G-TABULAR has resolved (the tabular library pick lives in
  `organize-ml-workspace` § G-TABULAR).
- G-PKG-NAME has resolved (`organize-ml-workspace` § G-PKG-NAME).

## The 9 steps (pixi)

### 1. Check pixi is on PATH

```bash
command -v pixi
```

If pixi is not installed, surface the install command and **ask the
user to run it** (do not run `curl | sh` yourself):

- Linux/macOS: `curl -fsSL https://pixi.sh/install.sh | sh`
- Windows: `iwr -useb https://pixi.sh/install.ps1 | iex`

### 2. Initialize the manifest

```bash
pixi init
```

Creates `pixi.toml` in the current directory. The default
generated content is minimal; the next step replaces it.

### 3. Apply the env layout

No user ask for extra envs. Edit `pixi.toml` with `default` and
`dev` only. Do not add an `agent` or `lsp` feature and do not add
`pyright`. `ipython` goes in `default` with the project stack so
the cell runner uses the same env.

```toml
[feature.default.dependencies]
# runtime deps land here (added in step 4), including ipython

[feature.dev.dependencies]
ruff = "*"
pytest = "*"
jupyterlab = "*"
ipykernel = "*"

[environments]
default = { features = ["default"],                       solve-group = "default" }
dev     = { features = ["default", "dev"],                solve-group = "default" }
```

### 4. Add Tier 1 runtime deps to `default`

The skore install variant follows G-SKORE-MODE. Pixi pulls from
conda-forge by default, so the `[jupyter]` extra is **not** needed
(it ships with the conda-forge package):

- `skore mode: local`  → `pixi add scikit-learn skrub skore`
- `skore mode: hub`    → `pixi add scikit-learn skrub "skore[hub]"`
- `skore mode: mlflow` → `pixi add scikit-learn skrub "skore[mlflow]" "mlflow>=3"`

The `mlflow` variant pins `mlflow>=3` explicitly: `skore[mlflow]`
alone can let the solver pick an old mlflow (2.x) the skore MLflow
backend doesn't support.

For PyPI-based managers running the analogous bootstrap (uv /
poetry / hatch / pip+venv), substitute `skore[jupyter]` (local),
`skore[hub,jupyter]` (hub), or `skore[mlflow,jupyter]` (mlflow) -
keeping the `mlflow>=3` pin for the mlflow variant. See SKILL.md
§ "Tier 1 install: skore variant per mode" for the full source-aware
table.

If G-SKORE-MODE hasn't fired yet at bootstrap time (rare -
`organize-ml-workspace` fires it alongside G-PKG-NAME and
G-TABULAR), route back to that skill before issuing the install
command. `ruff` / `pytest` / `jupyterlab` / `ipykernel` are added
by step 3 (the `[feature.dev]` declaration). `ipython` is added to
`default` with `pixi add ipython` when EDA or the audit will run.
No per-install ask. Do not add `pyright`.

**Graphviz:** do not install it in this bootstrap. EDA and the
baseline holdout do not need `dot`. Follow SKILL.md § "Graphviz"
only when a later plot or `draw_graph` / `full_report` asks for it.

### 5. Add the tabular library

Per G-TABULAR (`organize-ml-workspace`):

- pandas branch: `pixi add pandas pyarrow`
- polars branch: `pixi add polars`

### 6. Wire the editable workspace package

Per `references/editable_workspace.md`:

```bash
pixi add --pypi "<pkg> @ ."
```

Then edit `pixi.toml` `[pypi-dependencies]` to flip the entry to:

```toml
<pkg> = { path = ".", editable = true }
```

Then `pixi install`. The name-prefixed `<pkg> @ .` syntax is
required by pixi 0.69; the bare `--editable .` form is rejected
with `× URL requirement must be preceded by a package name`.

### 7. Sync default and dev

```bash
pixi install                # syncs default
pixi install -e dev
```

Do not create an `agent` or `lsp` env. Do not write
`pyrightconfig.json`.

### 8. Verify

```bash
pixi run python -c "import sklearn, skrub, skore, IPython"
pixi run -e dev ruff --version
```

Both commands must succeed before declaring the bootstrap complete.

## Other managers

If the user picked a different manager (uv / poetry / hatch /
conda), mirror this flow with:

- The manager's `init` command (uv `init`, poetry `init`, hatch
  `new`, conda `create -n <project>`).
- `ipython` in the same env as `skrub`, not a second env.
- The per-manager editable install from
  `references/editable_workspace.md`.

Do not add `pyright` and do not write `pyrightconfig.json`.
