#!/bin/sh
# Install arxiv-digest and add it to your applications. Run it again to update.
#
#   curl -LsSf https://raw.githubusercontent.com/SampsonML/arxiv-digest/main/install.sh | sh
#   ./install.sh               from a clone: install that clone's code instead
#   ... | sh -s -- --no-open   install without opening the app at the end
#
# It installs uv (https://docs.astral.sh/uv/) if needed, then arxiv-digest as a
# uv tool (`arxiv-digest` command in ~/.local/bin, its own Python environment),
# using the exact package versions in uv.lock.
#
# Environment: ARXIV_DIGEST_REF picks a branch or tag (default: main), and
# ARXIV_DIGEST_ARCHIVE a .tar.gz of the source to install instead (tests).

set -eu

REPO="SampsonML/arxiv-digest"
RAW_URL="https://raw.githubusercontent.com/$REPO/main/install.sh"

say() { printf '%s\n' "$*"; }
fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

download() { # download URL FILE
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$2" "$1"
  else
    fail "neither curl nor wget is available"
  fi
}

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

ensure_uv() {
  if uv=$(find_uv); then
    return 0
  fi
  say "arxiv-digest uses uv (https://docs.astral.sh/uv/) to set up Python and its packages."
  # When piped from curl, stdin is this script, so ask on the terminal instead.
  if (: </dev/tty) 2>/dev/null; then
    printf "Install uv now? [Y/n] "
    read -r answer </dev/tty || answer=""
    case "$answer" in
      [nN]*) fail "uv is needed. Install it yourself, or see \"Other ways to install\" in the README." ;;
    esac
  fi
  tmp_uv=$(mktemp)
  download "https://astral.sh/uv/install.sh" "$tmp_uv"
  sh "$tmp_uv"
  rm -f "$tmp_uv"
  uv=$(find_uv) || fail "uv was installed, but can't be found. Open a new terminal and run this again."
}

main() {
  open_flag=""
  for arg in "$@"; do
    case "$arg" in
      --no-open) open_flag="--no-open" ;;
      -h | --help)
        say "Install arxiv-digest and add it to your applications. Run it again to update."
        say "Options: --no-open (don't open the app at the end). See $RAW_URL"
        return 0
        ;;
      *) fail "unknown option: $arg" ;;
    esac
  done

  # From a clone (./install.sh), install the code next to this script.
  source_dir=""
  case "$0" in
    */install.sh | install.sh)
      here=$(cd "$(dirname "$0")" && pwd)
      if [ -f "$here/pyproject.toml" ] && grep -q '^name = "arxiv-digest"' "$here/pyproject.toml"; then
        source_dir=$here
      fi
      ;;
  esac

  ensure_uv

  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  trap 'exit 130' INT TERM
  if [ -z "$source_dir" ]; then
    ref=${ARXIV_DIGEST_REF:-main}
    archive=${ARXIV_DIGEST_ARCHIVE:-https://github.com/$REPO/archive/$ref.tar.gz}
    say "Downloading arxiv-digest ($ref) from GitHub..."
    case "$archive" in
      http://* | https://*) download "$archive" "$work/source.tar.gz" ;;
      *) cp "$archive" "$work/source.tar.gz" ;;
    esac
    mkdir "$work/source"
    tar -xzf "$work/source.tar.gz" -C "$work/source" --strip-components=1
    source_dir="$work/source"
  fi

  say "Installing arxiv-digest (the first time takes a minute)..."
  # The exact versions in uv.lock, so everyone gets the tested set of packages.
  "$uv" export --quiet --frozen --no-dev --no-hashes --no-emit-project \
    --directory "$source_dir" --output-file "$work/constraints.txt"
  "$uv" tool install --python ">=3.11,<3.15" --reinstall-package arxiv-digest \
    --constraints "$work/constraints.txt" "$source_dir"

  bin_dir=$("$uv" tool dir --bin)
  "$bin_dir/arxiv-digest" setup $open_flag

  say ""
  if [ "$source_dir" = "$work/source" ]; then
    say "To update arxiv-digest later, run the same command again:"
    say "  curl -LsSf $RAW_URL | sh"
  else
    say "To update later: git pull, then ./install.sh again."
  fi
  case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) say "To use the arxiv-digest command in a terminal, run \`uv tool update-shell\` and open a new terminal." ;;
  esac
}

main "$@"
