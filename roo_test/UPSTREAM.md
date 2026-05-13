# Roo-Eval upstream dependencies

Roo-Eval drives [Roo-Code](https://github.com/RooCodeInc/Roo-Code) (the
in-IDE agent) against exercises from
[Roo-Code-Evals](https://github.com/RooCodeInc/Roo-Code-Evals).

## Roo-Code (vendored, partial)

The Roo-Code source tree is vendored into `roo_test/Roo-Code/` at the
commit listed below, with build artifacts and dependency caches stripped
(`node_modules/`, `releases/`, `.turbo/`, `dist/`, `.next/`, `*.vsix`,
`.env*`, `.git/`). After cloning this repo, install Roo-Code's
dependencies before running evals:

```bash
cd "$CRANE_REPO_ROOT/roo_test/Roo-Code"
pnpm install         # requires Node 24+ and pnpm; respects pnpm-lock.yaml
pnpm build           # builds the VS Code extension + webview UI
```

Re-run `pnpm install` after pulling repo updates that touch
`pnpm-lock.yaml`.

## Roo-Code-Evals (referenced)

Roo-Code-Evals is the eval-exercise repo. We do **not** vendor it here
because it is independently versioned and updated frequently. Clone it
yourself before running any sweep:

```bash
cd "$CRANE_REPO_ROOT/roo_test"
git clone https://github.com/RooCodeInc/Roo-Code-Evals.git evals
git -C evals checkout 567d4aa9d2ce33d07c20a8c7e17450ee20152473
```

Override the location at run time via the YAML field `evals_repo`.

## Pinned commits (used in the paper's runs)

| Repo | URL | Commit |
|------|-----|--------|
| Roo-Code | https://github.com/RooCodeInc/Roo-Code.git | `7adbfec2a4219911be28b564986011e1088e5a6d` |
| Roo-Code-Evals | https://github.com/RooCodeInc/Roo-Code-Evals.git | `567d4aa9d2ce33d07c20a8c7e17450ee20152473` |

After cloning, follow `roo_test/README.md` to install Node 24, pnpm,
and the per-language toolchains (Go / Rust / Java) used by the eval
workspaces.
