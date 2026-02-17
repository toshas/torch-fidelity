#!/usr/bin/env bash
set -x
set -o pipefail

if ! type source > /dev/null 2>&1; then
    bash "$0" "$@"
    exit $?
fi

ROOT_DIR=$(realpath $(dirname "$0")/..)
RESULTS_DIR="${ROOT_DIR}/test_logs"
RESULTS_FILE="${RESULTS_DIR}/summary.txt"
rm -rf "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"
> "${RESULTS_FILE}"

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
        --rm --network=host --ipc=host \
        -v "${ROOT_DIR}":/work \
        --entrypoint "" \
        "torch-fidelity-test-${FLAVOR}" \
        "$@"
}

exec_cuda() {
    if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
        echo "CUDA_VISIBLE_DEVICES not set, using CUDA_VISIBLE_DEVICES=0"
        export CUDA_VISIBLE_DEVICES=0
    fi
    FLAVOR="${1}"
    shift
    cd ${ROOT_DIR} && docker run \
        --gpus "device=${CUDA_VISIBLE_DEVICES}" \
        --rm --network=host --ipc=host \
        --env CUDA_VISIBLE_DEVICES=0 \
        -v "${ROOT_DIR}":/work \
        --entrypoint "" \
        "torch-fidelity-test-${FLAVOR}" \
        "$@"
}

main() {
    FLAVOR="${1}"
    ARCH="${2}"
    WERR=""
    if [ ! -z "${3}" ]; then
        WERR="-W error"
    fi
    build "${FLAVOR}"
    exec_${ARCH} "${FLAVOR}" bash -c "cd /work && python3 ${WERR} -m unittest discover -s /work/tests/${FLAVOR} -t /work -p 'test_*.py'"
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
    # Inline docker run with -it for interactive shell (exec_cuda omits -it for scripted runs)
    cd ${ROOT_DIR} && docker run \
        --gpus "device=${CUDA_VISIBLE_DEVICES:-0}" \
        -it --rm --network=host --ipc=host \
        --env CUDA_VISIBLE_DEVICES=0 \
        -v "${ROOT_DIR}":/work \
        --entrypoint "" \
        "torch-fidelity-test-${FLAVOR}" \
        bash
}

record() {
    local label="${1}"
    shift
    local start=$SECONDS
    local log="${RESULTS_DIR}/${label}.log"
    if "$@" 2>&1 | tee "${log}"; then
        echo "PASS  $(($SECONDS - start))s  ${label}" >> "${RESULTS_FILE}"
    else
        echo "FAIL  $(($SECONDS - start))s  ${label}" >> "${RESULTS_FILE}"
    fi
}

run_all() {
    record torch_versions_ge_1_13_0  main torch_versions_ge_1_13_0 cuda werr
    if [ "${WITH_TF1}" = "1" ]; then
        record tf1                   main tf1 cuda               # fighting warnings in legacy/tensorflow is futile
    else
        echo "SKIP  0s  tf1 (use --with-tf1 to enable)" >> "${RESULTS_FILE}"
    fi
    record torch_pure                main torch_pure cuda werr
    record clip                      main clip cuda werr
    record prc_ppl_reference         main prc_ppl_reference cuda werr
    record sphinx_doc                main_sh sphinx_doc
}

# Parse args
WITH_TF1=0
for arg in "$@"; do
    case "$arg" in
        --with-tf1) WITH_TF1=1 ;;
    esac
done

run_all

echo ""
echo "==============================="
echo "  TEST RESULTS SUMMARY"
echo "==============================="
cat "${RESULTS_FILE}"
echo "==============================="

# Show individual test failures from logs
FAILURES=""
for log in "${RESULTS_DIR}"/*.log; do
    [ -f "${log}" ] || continue
    suite=$(basename "${log}" .log)
    # Grep for lines indicating individual test errors/failures (unittest output)
    fails=$(grep -E "^(FAIL|ERROR): " "${log}" 2>/dev/null || true)
    if [ -n "${fails}" ]; then
        FAILURES="${FAILURES}\n--- ${suite} ---\n${fails}\n"
    fi
done

if [ -n "${FAILURES}" ]; then
    echo "  FAILED TESTS:"
    echo "-------------------------------"
    echo -e "${FAILURES}"
    echo "==============================="
    echo "  Logs: ${RESULTS_DIR}/"
    echo "==============================="
fi

if grep -q "^FAIL" "${RESULTS_FILE}"; then
    exit 1
else
    exit 0
fi
