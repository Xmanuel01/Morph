#!/usr/bin/env bash
set -euo pipefail

REPO=${ENKAI_REPO:-${enkai_REPO:-Xmanuel01/Enkai}}
VERSION=${ENKAI_VERSION:-${enkai_VERSION:-latest}}

if ! command -v curl >/dev/null 2>&1; then
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

TMP=$(mktemp -d)
ARCHIVE="$TMP/$ASSET"
EXTRACT="$TMP/extract"
mkdir -p "$EXTRACT"

curl -fL "$URL" -o "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$EXTRACT"

INSTALL_DIR="${ENKAI_INSTALL_DIR:-${enkai_INSTALL_DIR:-$HOME/.local/bin}}"
mkdir -p "$INSTALL_DIR"

rm -rf "$INSTALL_DIR/std" "$INSTALL_DIR/examples"

cp "$EXTRACT/enkai" "$INSTALL_DIR/enkai"
chmod +x "$INSTALL_DIR/enkai"
if [ -f "$EXTRACT/enkai" ]; then
  cp "$EXTRACT/enkai" "$INSTALL_DIR/enkai"
  chmod +x "$INSTALL_DIR/enkai"
fi

if compgen -G "$EXTRACT/libenkai_native.*" > /dev/null; then
  cp "$EXTRACT"/libenkai_native.* "$INSTALL_DIR/"
fi
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

if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  PROFILE="$HOME/.profile"
  if [ -n "${ZSH_VERSION-}" ]; then
    PROFILE="$HOME/.zprofile"
  elif [ -n "${BASH_VERSION-}" ]; then
    PROFILE="$HOME/.bashrc"
  fi
  if ! grep -q "\.local/bin" "$PROFILE" 2>/dev/null; then
    echo "export PATH=\"$HOME/.local/bin:$PATH\"" >> "$PROFILE"
  fi
  echo "Added $INSTALL_DIR to PATH in $PROFILE. Restart your shell."
fi

"$INSTALL_DIR/enkai" --version

echo "Installed Enkai to $INSTALL_DIR"
