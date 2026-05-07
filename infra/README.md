# `infra/` — Vast.ai helpers

Local helper scripts used by the `/gpu` slash-command skill
(`~/.claude/commands/gpu.md`) and for direct vast.ai operations.

- `bootstrap_remote.sh` — one-command remote setup (pip installs, HF token
  copy, etc.) once an instance is reachable over SSH.
- `vast_gpu.sh` — convenience wrappers for vast.ai instance lifecycle (search,
  launch, destroy).

The skill itself is at `~/.claude/commands/gpu.md` and on GitHub at
`github.com/dafraile/claude-gpu-skill`.
