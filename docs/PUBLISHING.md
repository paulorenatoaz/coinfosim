# Publishing research reports

CoInfoSim publishes generated reports to the artifact-only `gh-pages` branch. The branch contains the academic homepage and generated `reports/` and, when enabled, `data/` content; it does not contain application source code.

## Normal publication

```bash
coinfosim publish
```

Normal publication is incremental: new and updated files are copied while previously published reports remain available. Scenario reports are discovered recursively and their internal links and images are validated before a commit is pushed.

## Validation without publishing

```bash
coinfosim publish --dry-run
```

This validates the output and assembles the homepage in a temporary directory without creating a worktree, branch, commit, or push.

## Publishing another output directory

```bash
coinfosim publish --output-dir ./experiments/example
```

## Exact synchronization

```bash
coinfosim publish --prune
```

`--prune` removes obsolete files from the managed `reports/` and `data/` directories before copying the current output. Unmanaged root files such as `CNAME` are preserved.

## One-time GitHub configuration

In the repository on GitHub, select:

```text
Settings
→ Pages
→ Build and deployment
→ Source: Deploy from a branch
→ Branch: gh-pages
→ Folder: /(root)
```

The public command creates and updates `gh-pages` automatically. It creates a commit only when the published content changes and pushes that commit to the configured remote.
