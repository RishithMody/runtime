@AGENTS.md

## Claude Code specifics

- Skills are in `.claude/skills/`. Load the matching one before a task in its area; the
  list is at the end of AGENTS.md.
- `.mcp.json` registers the `gitm` MCP server (`gitm mcp`: `list_runs`, `history`, `diff`).
  It needs the venv active, or `gitm` on PATH, with the `mcp` extra installed.
- For a broad question ("where is X decided"), `gitm/scheduler/loop.py` is the spine. Grep
  it for the phase first, then follow the call into `gitm/optimizer/`.
- Run `pytest tests/<file>.py` for the area you touched, then the full suite before
  saying you're done. Leave the regenerated `tests/fixtures/importers/*` files out of commits.
