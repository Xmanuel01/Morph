#!/usr/bin/env bash
set -euo pipefail

REPO="${ENKAI_REPO:-${enkai_REPO:-Xmanuel01/Enkai}}"
VERSION="${ENKAI_VERSION:-${enkai_VERSION:-latest}}"
INSTALL_DIR="${ENKAI_INSTALL_DIR:-${enkai_INSTALL_DIR:-$HOME/.local/bin}}"

if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "Error: curl or wget is required." >&2
  exit 1
fi

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
  linux) OS="linux" ;;
  darwin) OS="macos" ;;
  *)
    echo "Unsupported OS: $OS" >&2
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="aarch64" ;;
  *)
    echo "Unsupported architecture: $ARCH" >&2
    exit 1
    ;;
esac

ASSET="enkai-${VERSION}-${OS}-${ARCH}.tar.gz"
if [ "$VERSION" = "latest" ]; then
  BASE_URL="https://github.com/${REPO}/releases/latest/download"
else
  BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
fi

TMP_DIR="$(mktemp -d)"
ARCHIVE="${TMP_DIR}/${ASSET}"
CHECKSUM_URL="${ENKAI_CHECKSUM_URL:-${enkai_CHECKSUM_URL:-${BASE_URL}/${ASSET}.sha256}}"

echo "Downloading ${ASSET}..."
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${BASE_URL}/${ASSET}" -o "${ARCHIVE}"
else
  wget -q "${BASE_URL}/${ASSET}" -O "${ARCHIVE}"
fi

if [ "${ENKAI_SKIP_VERIFY:-${enkai_SKIP_VERIFY:-0}}" != "1" ]; then
  CHECKSUM_FILE="${TMP_DIR}/${ASSET}.sha256"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${CHECKSUM_URL}" -o "${CHECKSUM_FILE}" || true
  else
    wget -q "${CHECKSUM_URL}" -O "${CHECKSUM_FILE}" || true
  fi
  if [ -f "${CHECKSUM_FILE}" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
      (cd "${TMP_DIR}" && sha256sum -c "$(basename "${CHECKSUM_FILE}")")
    elif command -v shasum >/dev/null 2>&1; then
      EXPECTED="$(awk '{print $1}' "${CHECKSUM_FILE}")"
      ACTUAL="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
      if [ "${EXPECTED}" != "${ACTUAL}" ]; then
        echo "Checksum verification failed." >&2
        exit 1
      fi
    else
      echo "Warning: sha256sum not found; skipping checksum verification." >&2
    fi
  fi
fi

mkdir -p "${INSTALL_DIR}"
tar -xzf "${ARCHIVE}" -C "${TMP_DIR}"

if [ ! -f "${TMP_DIR}/enkai" ]; then
  echo "Error: enkai binary not found in archive." >&2
  exit 1
fi

mv -f "${TMP_DIR}/enkai" "${INSTALL_DIR}/enkai"
chmod +x "${INSTALL_DIR}/enkai"
if [ -f "${TMP_DIR}/enkai" ]; then
  mv -f "${TMP_DIR}/enkai" "${INSTALL_DIR}/enkai"
  chmod +x "${INSTALL_DIR}/enkai"
fi

if ! echo "$PATH" | tr ':' '\n' | grep -qx "${INSTALL_DIR}"; then
  PROFILE="${HOME}/.profile"
  if [ -n "${SHELL:-}" ] && [[ "${SHELL}" == *"zsh"* ]]; then
    PROFILE="${HOME}/.zprofile"
  fi
  echo "Adding ${INSTALL_DIR} to PATH in ${PROFILE}"
  printf '\nexport PATH="%s:$PATH"\n' "${INSTALL_DIR}" >> "${PROFILE}"
  echo "Restart your shell or run: source ${PROFILE}"
fi

echo "Enkai installed to ${INSTALL_DIR}/enkai"
echo "Verify: enkai --version"
