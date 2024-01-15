#!/usr/bin/env bash
set -e
set -x

ROOT_DIR=$(realpath $(dirname "$0")/..)

build() {
    FLAVOR="${1}"
    docker build \
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
    cd ${ROOT_DIR} && docker run \
        -it --rm --network=host --ipc=host \
        -v "${ROOT_DIR}":/work \
        "torch-fidelity-test-${FLAVOR}" \
        $@
}

exec_cuda() {
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        echo "CUDA_VISIBLE_DEVICES not set, using CUDA_VISIBLE_DEVICES=0"
        export CUDA_VISIBLE_DEVICES=0
    fi
    FLAVOR="${1}"
    shift
    cd ${ROOT_DIR} && nvidia-docker run \
        --gpus "device=${CUDA_VISIBLE_DEVICES}" \
        -it --rm --network=host --ipc=host \
        --env CUDA_VISIBLE_DEVICES=0 \
        -v "${ROOT_DIR}":/work \
        "torch-fidelity-test-${FLAVOR}" \
        $@
}

main() {
    FLAVOR="${1}"
    build "${FLAVOR}"
    exec_cuda "${FLAVOR}" python3 -W error -m unittest discover "tests/${FLAVOR}"
}

main_sh() {
    FLAVOR="${1}"
    build "${FLAVOR}"
    for test in tests/${FLAVOR}/test_*.sh; do
        exec_cuda "${FLAVOR}" sh ${test}
    done
}

shell() {
    FLAVOR="${1}"
    build "${FLAVOR}"
    exec_cuda "${FLAVOR}" bash
}

main tf1
main torch_pure
main clip
main prc_ppl_reference
main_sh sphinx_doc

echo "=== TESTS FINISHED ==="