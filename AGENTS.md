## Lean 4 Workflows

When a task involves `.lean` files, Lean 4 proofs, Lake builds, or mathlib search,
use the local Lean 4 skill in
`$HOME/LeanRace/tools/lean4-skills/plugins/lean4/skills/lean4/SKILL.md`.

Before running the skill's helper scripts, source the local environment shim:

```bash
source "$HOME/LeanRace/.agents/lean4-env.sh"
```

Environment:
- `LEAN4_PLUGIN_ROOT=$HOME/LeanRace/tools/lean4-skills/plugins/lean4`
- `LEAN4_SCRIPTS=$HOME/LeanRace/tools/lean4-skills/plugins/lean4/lib/scripts`
- `LEAN4_REFS=$HOME/LeanRace/tools/lean4-skills/plugins/lean4/skills/lean4/references`
