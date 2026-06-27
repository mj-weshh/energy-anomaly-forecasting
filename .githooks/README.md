# Git hooks

## This repository

After cloning, enable project hooks once:

```powershell
.\scripts\setup-git-hooks.ps1
```

Or manually:

```powershell
git config core.hooksPath .githooks
```

The `commit-msg` hook silently removes `Co-authored-by: Cursor <cursoragent@cursor.com>` from every commit message.

## All repositories (global template)

One-time user setup so new repos inherit the same hook:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.git-template\hooks"
Copy-Item .githooks\commit-msg "$env:USERPROFILE\.git-template\hooks\commit-msg"
git config --global init.templateDir "$env:USERPROFILE\.git-template"
```

For **existing** clones of other repos, either run `git config core.hooksPath .githooks` in each repo (if you copy the hook there) or copy `commit-msg` into that repo's `.git/hooks/` directory.

## Cursor IDE (optional)

In **Cursor Settings**, search for co-author or attribution options and disable automatic co-author injection on agent commits. Wording varies by Cursor version. Hooks are the safety net if the IDE still adds the trailer.
