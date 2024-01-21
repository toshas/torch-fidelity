#!/usr/bin/env bash
set -e
set -x

if ! type source > /dev/null 2>&1; then
    bash "$0" "$@"
    exit $?
fi

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
    ARCH="${2}"
    WERR=""
    if [ ! -z "${3}" ]; then
        WERR="-W error"
    fi
    build "${FLAVOR}"
    exec_${ARCH} "${FLAVOR}" python3 ${WERR} -m unittest discover "tests/${FLAVOR}"
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

run_all() {
    time main torch_versions_ge_1_11_0 cuda werr
    time main tf1 cuda               # fighting warnings in legacy/tensorflow is futile
    time main torch_pure cuda werr
    time main clip cuda werr
    time main prc_ppl_reference cuda werr
    time main_sh sphinx_doc
}

time run_all
echo "=== TESTS FINISHED ==="