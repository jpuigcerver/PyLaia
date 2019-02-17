#!/usr/bin/env bash

function get_num_cores () {
  nproc 2> /dev/null || echo 1;
  return 0;
}

function wait_jobs () {
  local n=0;
  local log_dir=;
  if [ "$1" = "--log_dir" ]; then
    log_dir="$1";
    shift 1;
  fi;
  while [ $# -gt 0 ]; do
    if ! wait "$1"; then
      echo "Failed subprocess with PID ${1}." >&2;
      if [ -n "$log_dir" ]; then
	echo "Dumping content of ${log_dir}/$n..." >&2;
	cat "${log_dir}/$n" >&2;
      fi;
      return 1;
    fi;
    shift 1; ((++n));
  done;
  return 0;
}
