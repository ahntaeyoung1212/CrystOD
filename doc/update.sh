#!/usr/bin/env bash
#
# Build the CrystOD documentation and publish it to the laboratory site.
#
#   ~/CrystOD-main/doc              Sphinx sources (this directory)
#   ~/mochizuki-tus.github.io       the GitHub Pages repository
#                 └── CrystOD/      where the built HTML is published
#
# Usage:
#   ./update.sh                       build, publish, commit with a default message
#   ./update.sh "docs: fix the logo"  ... with your own commit message
#   ./update.sh --no-push "msg"       do everything but the final push
#   ./update.sh --dry-run             build and show what WOULD be published
#
set -euo pipefail

DOC_DIR="$HOME/CrystOD-main/doc"
SITE_REPO="$HOME/mochizuki-tus.github.io"
SITE_SUBDIR="CrystOD"
CONDA_ENV="crystod"

DRY_RUN=0
PUSH=1
MESSAGE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-push) PUSH=0 ;;
    -h|--help) sed -n '3,14p' "$0"; exit 0 ;;
    *) MESSAGE="$1" ;;
  esac
  shift
done
: "${MESSAGE:=docs: rebuild the CrystOD site}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
die() { printf '\n\033[1;31mError: %s\033[0m\n' "$1" >&2; exit 1; }

[ -d "$SITE_REPO/.git" ] || die "$SITE_REPO is not a git repository."

# --- 1. environment -----------------------------------------------------------
# `conda activate` is a shell function, so the hook has to be sourced first --
# it is not available in a non-interactive script otherwise.
say "Activating the '$CONDA_ENV' conda environment"
# shellcheck disable=SC1091
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- 2. build -----------------------------------------------------------------
say "Building the HTML"
cd "$DOC_DIR"
make html

# --- 3. publish into the site repository --------------------------------------
# --delete keeps the published tree free of files a rebuild no longer produces
# (stale theme assets, renamed pages).  Two things must survive it:
#   sitemap.xml   hand-written, Sphinx does not generate it
#   .buildinfo    a build artefact with no place in the published site
say "Syncing _build/html -> $SITE_REPO/$SITE_SUBDIR"
RSYNC_FLAGS=(-a --delete --exclude='sitemap.xml' --exclude='.buildinfo*')
if [ "$DRY_RUN" -eq 1 ]; then
  RSYNC_FLAGS+=(--dry-run --itemize-changes)
fi
rsync "${RSYNC_FLAGS[@]}" "$DOC_DIR/_build/html/" "$SITE_REPO/$SITE_SUBDIR/"

if [ "$DRY_RUN" -eq 1 ]; then
  say "Dry run: nothing was written, nothing committed."
  exit 0
fi

# --- 4. commit ----------------------------------------------------------------
# Staged deliberately narrowly: the site repository often carries unrelated
# work in progress at its root (index.html and friends), which must not be
# swept into a documentation commit.
say "Staging $SITE_SUBDIR/"
cd "$SITE_REPO"
git add "$SITE_SUBDIR"

if git diff --cached --quiet; then
  say "No changes to publish -- the site is already up to date."
  exit 0
fi

git status --short -- "$SITE_SUBDIR"
say "Committing: $MESSAGE"
git commit -m "$MESSAGE"

# --- 5. push ------------------------------------------------------------------
if [ "$PUSH" -eq 1 ]; then
  say "Pushing to $(git remote get-url origin)"
  git push
  say "Published. https://mochizuki-tus.github.io/CrystOD/ updates in a minute or two."
else
  say "Committed but not pushed (--no-push).  Run 'git push' in $SITE_REPO when ready."
fi
