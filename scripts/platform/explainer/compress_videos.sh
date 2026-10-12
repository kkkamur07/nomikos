#!/usr/bin/env bash
# Compress the recorded explainer videos for the landing page.
#
# Usage: compress_videos.sh SOURCE_DIR [OUT_DIR]
#   SOURCE_DIR  folder holding nomikos-whats-new.mp4 and nomikos-explainer.mp4
#   OUT_DIR     default: landing/assets/video in this checkout
#
# For each <name>.mp4 it writes <name>.mp4 (H.264), <name>.webm (VP9) and
# <name>-poster.webp (first frame after 1 s), 960 px wide at 24 fps.
# Defaults (mp4 crf 35, VP9 crf 44 tuned for screen content) are the smallest
# settings where the caption bar and toolbar labels stay legible at 960 wide.
# Re-running skips a video whose three outputs are newer than both its source
# and this script (VP9 is not bit-exact across runs); FORCE=1 re-encodes.
# MP4_CRF and WEBM_CRF override the defaults (used to compare quality levels,
# together with FORCE=1).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SOURCE_DIR [OUT_DIR]" >&2
  exit 2
fi

FFMPEG="$(command -v ffmpeg || true)"
CWEBP="$(command -v cwebp || true)"
if [[ -z "$FFMPEG" || -z "$CWEBP" ]]; then
  echo "ffmpeg and cwebp must be on PATH (brew install ffmpeg webp)" >&2
  exit 1
fi

SRC_DIR="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${2:-$REPO_ROOT/landing/assets/video}"
MP4_CRF="${MP4_CRF:-35}"
WEBM_CRF="${WEBM_CRF:-44}"
NAMES=(nomikos-whats-new nomikos-explainer)
VF="scale=960:-2:flags=lanczos,fps=24,format=yuv420p"

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for name in "${NAMES[@]}"; do
  src="$SRC_DIR/$name.mp4"
  if [[ ! -f "$src" ]]; then
    echo "missing $src" >&2
    exit 1
  fi

  outs=("$OUT_DIR/$name.mp4" "$OUT_DIR/$name.webm" "$OUT_DIR/$name-poster.webp")
  fresh=1
  for out in "${outs[@]}"; do
    if [[ ! "$out" -nt "$src" || ! "$out" -nt "${BASH_SOURCE[0]}" ]]; then
      fresh=0
    fi
  done
  if [[ "${FORCE:-0}" != 1 && "$fresh" == 1 ]]; then
    echo "$name: up to date, skipped"
    continue
  fi

  "$FFMPEG" -hide_banner -loglevel error -y -i "$src" -an -vf "$VF" \
    -c:v libx264 -preset slow -tune animation -crf "$MP4_CRF" \
    -movflags +faststart "$OUT_DIR/$name.mp4"

  "$FFMPEG" -hide_banner -loglevel error -y -i "$src" -an -vf "$VF" \
    -c:v libvpx-vp9 -crf "$WEBM_CRF" -b:v 0 -row-mt 1 -deadline good -cpu-used 2 \
    -tune-content screen -g 240 \
    "$OUT_DIR/$name.webm"

  # ffmpeg builds often lack libwebp, so go through a PNG and cwebp.
  tmp_png="$TMP_DIR/$name-poster.png"
  "$FFMPEG" -hide_banner -loglevel error -y -ss 1 -i "$src" -frames:v 1 \
    -vf "scale=960:-2:flags=lanczos" "$tmp_png"
  "$CWEBP" -quiet -q 80 "$tmp_png" -o "$OUT_DIR/$name-poster.webp"

  echo "$name: $(du -h "$OUT_DIR/$name.mp4" | cut -f1) mp4, $(du -h "$OUT_DIR/$name.webm" | cut -f1) webm"
done
