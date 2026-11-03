#!/bin/sh
# Install arxiv-digest from this folder and add it to your applications.
# Run it again after `git pull` to update.
#
#   ./install.sh             install, add the app, and open it
#   ./install.sh --no-open   the same, without opening the browser
#
# It uses uv (https://docs.astral.sh/uv/) to set up a supported Python and the
# packages in .venv inside this folder; if uv is missing, it offers to install it.
set -eu

cd "$(dirname "$0")"

for arg in "$@"; do
  case "$arg" in
    --no-open) ;;
    -h | --help)
      sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (see ./install.sh --help)" >&2
      exit 2
      ;;
  esac
done

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi
  # Where uv's installer puts it, for when this shell's PATH doesn't have it yet.
  for candidate in "${XDG_BIN_HOME:-}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! uv=$(find_uv); then
  echo "arxiv-digest uses uv (https://docs.astral.sh/uv/) to set up Python and its packages."
  if [ -t 0 ]; then
    printf "Install uv now? [Y/n] "
    read -r answer || answer=n
    case "$answer" in
      [nN]*)
        echo "Install uv yourself, or see \"Other ways to install\" in README.md."
        exit 1
        ;;
    esac
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "Neither curl nor wget is available. Install uv from https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  if ! uv=$(find_uv); then
    echo "uv was installed, but this script can't find it. Open a new terminal and run ./install.sh again." >&2
    exit 1
  fi
fi

echo "Setting up arxiv-digest in $(pwd)/.venv (the first time takes a minute)..."
"$uv" sync --locked
exec "$(pwd)/.venv/bin/python" -m arxiv_digest setup "$@"
