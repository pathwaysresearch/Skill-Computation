#!/usr/bin/env bash
set -euo pipefail

MODE="${APP_MODE:-streamlit}"
PORT="${PORT:-8080}"

export ISB_IGRAPH_RUNTIME_ROOT="${ISB_IGRAPH_RUNTIME_ROOT:-/app/runtime}"
export ISB_IGRAPH_UPLOADS_ROOT="${ISB_IGRAPH_UPLOADS_ROOT:-${ISB_IGRAPH_RUNTIME_ROOT}/uploads}"
export ISB_IGRAPH_ARTIFACTS_ROOT="${ISB_IGRAPH_ARTIFACTS_ROOT:-${ISB_IGRAPH_RUNTIME_ROOT}/artifacts}"
export ISB_IGRAPH_JOBS_DB="${ISB_IGRAPH_JOBS_DB:-${ISB_IGRAPH_RUNTIME_ROOT}/jobs.db}"

mkdir -p "${ISB_IGRAPH_UPLOADS_ROOT}" "${ISB_IGRAPH_ARTIFACTS_ROOT}"

run_streamlit() {
  exec streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT}"
}

run_api() {
  exec uvicorn services.api.main:app --host 0.0.0.0 --port "${PORT}"
}

run_worker() {
  exec python -m services.worker.main --poll-interval "${WORKER_POLL_INTERVAL:-2}"
}

run_all_in_one() {
  export ISB_IGRAPH_API_BASE_URL="${ISB_IGRAPH_API_BASE_URL:-http://127.0.0.1:8000}"

  uvicorn services.api.main:app --host 0.0.0.0 --port 8000 &
  api_pid=$!

  python -m services.worker.main --poll-interval "${WORKER_POLL_INTERVAL:-2}" &
  worker_pid=$!

  streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT}" &
  streamlit_pid=$!

  cleanup() {
    kill "${api_pid}" "${worker_pid}" "${streamlit_pid}" 2>/dev/null || true
    wait "${api_pid}" "${worker_pid}" "${streamlit_pid}" 2>/dev/null || true
  }

  trap cleanup INT TERM

  wait -n "${api_pid}" "${worker_pid}" "${streamlit_pid}"
  status=$?
  cleanup
  exit "${status}"
}

case "${MODE}" in
  streamlit)
    run_streamlit
    ;;
  api)
    run_api
    ;;
  worker)
    run_worker
    ;;
  all-in-one)
    run_all_in_one
    ;;
  *)
    echo "Unknown APP_MODE='${MODE}'. Use one of: streamlit, api, worker, all-in-one." >&2
    exit 1
    ;;
esac
