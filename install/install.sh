#!/usr/bin/env bash
set -euo pipefail

REPO=${ENKAI_REPO:-${enkai_REPO:-Xmanuel01/Enkai}}
VERSION=${ENKAI_VERSION:-${enkai_VERSION:-latest}}
BUNDLE_PATH=""
INSTALL_DIR="${ENKAI_INSTALL_DIR:-${enkai_INSTALL_DIR:-$HOME/.local/bin}}"
NO_PATH_UPDATE=0
UNINSTALL=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --bundle-path) BUNDLE_PATH="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --no-path-update) NO_PATH_UPDATE=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

managed_entries() {
  printf '%s\n' "enkai" "README.txt" "bundle_manifest.json" "std" "examples" "install_manifest.json"
  for name in "$INSTALL_DIR"/libenkai_native.*; do
    [ -e "$name" ] && basename "$name"
  done
}

remove_managed_entries() {
  mkdir -p "$INSTALL_DIR"
  while IFS= read -r entry; do
    [ -z "$entry" ] && continue
    rm -rf "$INSTALL_DIR/$entry"
  done <<EOF
$(managed_entries)
EOF
  if [ -d "$INSTALL_DIR" ] && [ -z "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
    rmdir "$INSTALL_DIR"
  fi
}

write_install_manifest() {
  local version="$1"
  local source_type="$2"
  local source_value="$3"
  python3 - "$INSTALL_DIR/install_manifest.json" "$version" "$source_type" "$source_value" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "installed_version": sys.argv[2],
    "source_type": sys.argv[3],
    "source_value": sys.argv[4],
    "managed_entries": [
        "enkai",
        "README.txt",
        "bundle_manifest.json",
        "std",
        "examples",
        "install_manifest.json",
    ],
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

get_installed_version() {
  local output
  output=$("$INSTALL_DIR/enkai" --version)
  printf "%s\n" "$output" | sed -nE 's/^Enkai v([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' | head -n1
}

test_bundle_manifest_version() {
  if [ ! -f "$INSTALL_DIR/bundle_manifest.json" ]; then
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$INSTALL_DIR/bundle_manifest.json" "$1" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
actual = payload.get("version")
if actual and actual != expected:
    raise SystemExit(f"Bundle manifest version {actual} does not match installed binary version {expected}")
PY
  fi
}

if [ "$UNINSTALL" -eq 1 ]; then
  remove_managed_entries
  echo "Uninstalled Enkai from $INSTALL_DIR"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1 && [ -z "$BUNDLE_PATH" ]; then
  echo "curl is required" >&2
  exit 1
fi

OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
  Linux) OS=linux ;;
  Darwin) OS=macos ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

case "$ARCH" in
  x86_64|amd64) ARCH=x86_64 ;;
  arm64|aarch64) ARCH=aarch64 ;;
  *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

TMP=$(mktemp -d)
cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT

if [ -n "$BUNDLE_PATH" ]; then
  ARCHIVE="$BUNDLE_PATH"
  SOURCE_TYPE="local_bundle"
  SOURCE_VALUE="$BUNDLE_PATH"
  if [ ! -f "$ARCHIVE" ]; then
    echo "Bundle not found: $ARCHIVE" >&2
    exit 1
  fi
  TAG="$VERSION"
  VER=${TAG#v}
else
  if [ "$VERSION" = "latest" ]; then
    API_URL="https://api.github.com/repos/$REPO/releases/latest"
  else
    API_URL="https://api.github.com/repos/$REPO/releases/tags/$VERSION"
  fi

  JSON=$(curl -fsSL "$API_URL")
  TAG=$(printf "%s" "$JSON" | grep -m1 '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
  if [ -z "$TAG" ]; then
    echo "Failed to determine release tag" >&2
    exit 1
  fi
  VER=${TAG#v}
  ASSET="enkai-${VER}-${OS}-${ARCH}.tar.gz"
  URL=$(printf "%s" "$JSON" | grep -E '"browser_download_url"' | grep "$ASSET" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')
  if [ -z "$URL" ]; then
    echo "Asset not found: $ASSET" >&2
    exit 1
  fi
  CHECKSUM_URL=$(printf "%s" "$JSON" | grep -E '"browser_download_url"' | grep "${ASSET}\.sha256" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/')

  ARCHIVE="$TMP/$ASSET"
  curl -fL "$URL" -o "$ARCHIVE"
  if [ -n "$CHECKSUM_URL" ]; then
    CHECKSUM_FILE="$TMP/$ASSET.sha256"
    curl -fL "$CHECKSUM_URL" -o "$CHECKSUM_FILE"
    if command -v sha256sum >/dev/null 2>&1; then
      (cd "$TMP" && sha256sum -c "$(basename "$CHECKSUM_FILE")")
    elif command -v shasum >/dev/null 2>&1; then
      EXPECTED=$(cut -d' ' -f1 "$CHECKSUM_FILE")
      ACTUAL=$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)
      [ "$EXPECTED" = "$ACTUAL" ] || { echo "Checksum mismatch for $ASSET" >&2; exit 1; }
    fi
  fi
  SOURCE_TYPE="github_release"
  SOURCE_VALUE="$URL"
fi

EXTRACT="$TMP/extract"
mkdir -p "$EXTRACT"
tar -xzf "$ARCHIVE" -C "$EXTRACT"

mkdir -p "$INSTALL_DIR"
remove_managed_entries
mkdir -p "$INSTALL_DIR"

cp "$EXTRACT/enkai" "$INSTALL_DIR/enkai"
chmod +x "$INSTALL_DIR/enkai"

if compgen -G "$EXTRACT/libenkai_native.*" > /dev/null; then
  cp "$EXTRACT"/libenkai_native.* "$INSTALL_DIR/"
fi
if [ -d "$EXTRACT/std" ]; then
  cp -R "$EXTRACT/std" "$INSTALL_DIR/std"
fi
if [ -d "$EXTRACT/examples" ]; then
  cp -R "$EXTRACT/examples" "$INSTALL_DIR/examples"
fi
if [ -f "$EXTRACT/README.txt" ]; then
  cp "$EXTRACT/README.txt" "$INSTALL_DIR/README.txt"
fi
if [ -f "$EXTRACT/bundle_manifest.json" ]; then
  cp "$EXTRACT/bundle_manifest.json" "$INSTALL_DIR/bundle_manifest.json"
fi

INSTALLED_VERSION=$(get_installed_version)
if [ -z "$INSTALLED_VERSION" ]; then
  echo "Failed to parse installed Enkai version" >&2
  exit 1
fi
test_bundle_manifest_version "$INSTALLED_VERSION"
write_install_manifest "$INSTALLED_VERSION" "$SOURCE_TYPE" "$SOURCE_VALUE"

if [ "$NO_PATH_UPDATE" -ne 1 ]; then
  if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    PROFILE="$HOME/.profile"
    if [ -n "${ZSH_VERSION-}" ]; then
      PROFILE="$HOME/.zprofile"
    elif [ -n "${BASH_VERSION-}" ]; then
      PROFILE="$HOME/.bashrc"
    fi
    if ! grep -q "$INSTALL_DIR" "$PROFILE" 2>/dev/null; then
      echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$PROFILE"
    fi
    echo "Added $INSTALL_DIR to PATH in $PROFILE. Restart your shell."
  fi
fi

"$INSTALL_DIR/enkai" --version
echo "Installed Enkai to $INSTALL_DIR"
