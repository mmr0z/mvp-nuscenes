#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v gh >/dev/null || { echo 'Install GitHub CLI (gh), then run gh auth login.' >&2; exit 1; }
gh auth status
OWNER="$(gh api user --jq .login)"
[[ "$OWNER" == mmr0z ]] || { echo "Expected account mmr0z, got $OWNER" >&2; exit 1; }
REPO="$OWNER/mvp-training"
python3 scripts/verify_models.py
if gh repo view "$REPO" --json isPrivate >/dev/null 2>&1; then
  [[ "$(gh repo view "$REPO" --json isPrivate --jq .isPrivate)" == true ]] || { echo 'Existing repository is public; refusing to upload.' >&2; exit 1; }
else
  gh repo create "$REPO" --private --description 'MVP: CenterNet2 + CenterPoint, training, validation, tests and checkpoints'
fi
[[ "$(gh repo view "$REPO" --json isPrivate --jq .isPrivate)" == true ]] || exit 1
gh auth setup-git
if git remote get-url origin >/dev/null 2>&1; then
  [[ "$(git remote get-url origin)" == "https://github.com/$REPO.git" ]] || { echo 'Unexpected origin; refusing to push.' >&2; exit 1; }
else
  git remote add origin "https://github.com/$REPO.git"
fi
git push -u origin main
if ! gh release view models-v1 --repo "$REPO" >/dev/null 2>&1; then
  gh release create models-v1 --repo "$REPO" --target main --title 'MVP model checkpoints' --notes 'CenterNet2 and trained MVP CenterPoint epoch 6. SHA256 checksums: models/manifest.json.'
fi
gh release upload models-v1 models/centernet2_checkpoint.pth models/mvp_centerpoint_epoch_6.pth --repo "$REPO" --clobber
gh repo view "$REPO" --json url,isPrivate
