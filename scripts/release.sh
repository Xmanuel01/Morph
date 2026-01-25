#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/release.sh vX.Y.Z [--skip-tests] [--push] [--allow-dirty]

Examples:
  scripts/release.sh v0.8.0
  scripts/release.sh v0.8.0 --skip-tests
  scripts/release.sh v0.8.0 --push

Notes:
  - Requires clean git status unless --allow-dirty is set.
  - Creates an annotated tag "vX.Y.Z".
  - Pushes to origin only if --push is provided.
EOF
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

VERSION="$1"
shift

SKIP_TESTS=0
DO_PUSH=0
ALLOW_DIRTY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-tests) SKIP_TESTS=1 ;;
    --push) DO_PUSH=1 ;;
    --allow-dirty) ALLOW_DIRTY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if ! [[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version must be in vX.Y.Z format (e.g., v0.8.0)" >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "main" ]; then
  echo "Warning: you are on branch '$BRANCH' (expected 'main')." >&2
fi

if [ "$ALLOW_DIRTY" -ne 1 ]; then
  if [ -n "$(git status --porcelain)" ]; then
    echo "Working tree is dirty. Commit or stash changes (or use --allow-dirty)." >&2
    exit 1
  fi
fi

if [ "$SKIP_TESTS" -ne 1 ]; then
  echo "Running format + clippy + tests..."
  cargo fmt
  cargo clippy -- -D warnings
  cargo test
fi

TAG_MSG="Release ${VERSION}"
if git rev-parse "$VERSION" >/dev/null 2>&1; then
  echo "Tag $VERSION already exists. Aborting." >&2
  exit 1
fi

git tag -a "$VERSION" -m "$TAG_MSG"
echo "Created tag $VERSION"

if [ "$DO_PUSH" -eq 1 ]; then
  git push origin "$BRANCH"
  git push origin "$VERSION"
  echo "Pushed $BRANCH and $VERSION to origin."
else
  echo "To push:"
  echo "  git push origin $BRANCH"
  echo "  git push origin $VERSION"
fi
