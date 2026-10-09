# Hub credentials from `.skore` (this lab)

This lab's coding agent **never** uses interactive hub login (`skore
login`, browser OAuth, device-code prompts, `input()`, `getpass`, or
`skore agent`). `skore agent --harness bob-ide` would also write
`.bob/mcp.json`, which this lab must not install.

Participants run `python scripts/install_skore.py` once
(see the lab `GUIDED.md`). That writes gitignored `.skore` JSON
(`hub_url`, `workspace`, `workspace_id`, `api_key`).

**Tool:** `load_skore_credentials()` in
`scripts/load_skore_credentials.py` (this skill). Scaffold copies
`templates/src_hub.py` to `src/<pkg>/hub.py`. It finds the nearest
`.skore` (walk up from CWD, then from the module). Reuse this
snippet every time - do not invent another loader, do not wrap
`login`, do not set `SKORE_HUB_URI` / `SKORE_HUB_API_KEY` by hand.

`skore.login()` reads those env vars. If they are set, it uses the
API key and **does not** open a browser.

## Required snippet (reuse byte-identical before every hub `login`)

```python
from skore import Project, login

from <pkg>.hub import load_skore_credentials

cfg = load_skore_credentials()
login(mode="hub")
project = Project(
    name="ibm-hackaton",
    mode="hub",
    workspace=cfg["workspace"],
)
```

Always `load_skore_credentials()` then `login(mode="hub")`, in that
order, in the same cell.

CLI check (prints `hub_url` and `workspace`, never the key):

```bash
python skills/organize-ml-workspace/scripts/load_skore_credentials.py
```

## Forbidden

- `skore login` (CLI) and `skore agent` / `--harness bob-ide` (those
  launch Bob or MCP). Credentials come only from `.skore`.
- Calling `login()` without `load_skore_credentials()` first - that
  falls back to interactive `Token` / browser.
- A `login_hub()` wrapper or copy-pasted `os.environ["SKORE_HUB_URI"]`.
  Import `<pkg>.hub.load_skore_credentials`.
- Asking the user to pick a workspace. An API key belongs to
  exactly one; read `workspace` from `.skore`.
- Pointing `SKORE_HUB_URI` at the public hub
  (`https://api.skore.probabl.ai`) when `.skore` has the lab API URL.

If `login` fails: check that `.skore` exists and that
`scripts/install_skore.py` was run. Do not retry with a
browser flow.
