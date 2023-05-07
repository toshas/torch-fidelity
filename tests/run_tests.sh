#!/usr/bin/env bash
set -e
set -x

ROOT_DIR=$(realpath $(dirname "$0")/..)

build() {
    FLAVOR="${1}"
    nvidia-docker build \
        --pull \
        --build-arg UNAME=$(whoami) \
        --build-arg UID=$(id -u) \
        --build-arg GID=$(id -g) \
        --tag "torch-fidelity-test-${FLAVOR}" \
        -f "${ROOT_DIR}/tests/${FLAVOR}/Dockerfile" \
        "${ROOT_DIR}"
}

exec_cpu() {
    FLAVOR="${1}"
    shift
    cd ${ROOT_DIR} && nvidia-docker run \
        -it --rm --network=host \
        -v "${ROOT_DIR}":/work \
        "torch-fidelity-test-${FLAVOR}" \
        $@
}

exec_cuda() {
    FLAVOR="${1}"
    shift
    cd ${ROOT_DIR} && nvidia-docker run \
        -it --rm --network=host \
        --env CUDA_VISIBLE_DEVICES=0 \
        -v "${ROOT_DIR}":/work \
        "torch-fidelity-test-${FLAVOR}" \
        $@
}

main() {
    FLAVOR="${1}"
    build "${FLAVOR}"
    exec_cpu "${FLAVOR}" python3 -m unittest discover "tests/${FLAVOR}"
}

shell() {
    FLAVOR="${1}"
    build "${FLAVOR}"
    exec_cuda "${FLAVOR}" bash
}

main tf1 || true
main torch_pure || true
main clip || true
main backend || true
main prc_ppl_reference || true
