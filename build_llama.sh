#!/usr/bin/env bash
# Thin wrapper around llama-infer-opt build + optional model download.
#
# Options (positional or env):
#   --with-model              also run setup-models.sh after build
#   --model-only              skip build, only download the model
#
# Pass-through env vars supported by the underlying build script:
#   BACKEND=cuda|metal|cpu   (auto-detected if unset)
#   CUDA_ARCH=86|89|...
#   BUILD_DIR=build          (relative to llama repo)
#   JOBS=<n>
#   TARGETS="llama-server"   (default: llama-server llama-mtmd-cli)
#
# Env vars forwarded to setup-models.sh:
#   MODELS_DIR=$HOME/models
#   MODELS="qwen25vl"
#
# Optional: LLAMA_DIR to point at a different llama.cpp checkout.
#
# Examples:
#   ./build_llama.sh
#   ./build_llama.sh --with-model
#   MODELS_DIR=./models ./build_llama.sh --with-model
#   ./build_llama.sh --model-only

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HERE/../llama-infer-opt}"

WITH_MODEL=0
MODEL_ONLY=0
passthrough=()
for arg in "$@"; do
    case "$arg" in
        --with-model) WITH_MODEL=1 ;;
        --model-only) MODEL_ONLY=1; WITH_MODEL=1 ;;
        *) passthrough+=("$arg") ;;
    esac
done

if [[ ! -x "$LLAMA_DIR/scripts/build.sh" ]]; then
    echo "[build_llama] cannot find $LLAMA_DIR/scripts/build.sh" >&2
    echo "[build_llama] set LLAMA_DIR to your llama.cpp checkout" >&2
    exit 1
fi

echo "[build_llama] using LLAMA_DIR=$LLAMA_DIR"

if [[ "$MODEL_ONLY" -eq 0 ]]; then
    "$LLAMA_DIR/scripts/build.sh" "${passthrough[@]}"

    BUILD_DIR="${BUILD_DIR:-build}"
    BIN="$LLAMA_DIR/$BUILD_DIR/bin/llama-server"
    if [[ -x "$BIN" ]]; then
        echo
        echo "[build_llama] llama-server built at:"
        echo "    $BIN"
    fi
fi

if [[ "$WITH_MODEL" -eq 1 ]]; then
    if [[ ! -x "$LLAMA_DIR/scripts/setup-models.sh" ]]; then
        echo "[build_llama] cannot find $LLAMA_DIR/scripts/setup-models.sh" >&2
        exit 1
    fi
    echo
    echo "[build_llama] downloading model(s) ..."
    "$LLAMA_DIR/scripts/setup-models.sh"

    MODELS_DIR="${MODELS_DIR:-$HOME/models}"
    GGUF="$(find "$MODELS_DIR" -maxdepth 2 -type f -name '*.gguf' ! -name 'mmproj*' | head -1)"
    if [[ -n "${GGUF:-}" ]]; then
        echo
        echo "[build_llama] launch RAG stack with:"
        echo "    python3 run_rapid_llama.py \\"
        echo "      --llama-server $LLAMA_DIR/${BUILD_DIR:-build}/bin/llama-server \\"
        echo "      --model $GGUF"
    fi
fi
