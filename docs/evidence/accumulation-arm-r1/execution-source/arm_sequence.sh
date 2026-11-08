#!/usr/bin/env bash
# Bounded sequential driver for the predeclared accumulation arms.
# One heavy job at a time; waits for OS commit room before each admission and
# renames (never deletes) a pressure-refused receipt before retrying.
set -u
cd "$(dirname "$0")/.." || exit 1
P="$(pwd -W)/.venv/Scripts/python.exe"
STOP=31368351744
LOG=runs/arm-sequence.out
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

commit_bytes() {
  powershell -NoProfile -Command "(Get-Counter '\Memory\Committed Bytes').CounterSamples.CookedValue" | tr -d '\r'
}

wait_room() {
  local need_mib=$1
  for _ in $(seq 1 1800); do
    local c
    c=$(commit_bytes)
    if "$P" -c "import sys;sys.exit(0 if int('$c')+int($need_mib)*1048576+67108864 < $STOP else 1)"; then
      log "room ok for ${need_mib}MiB at commit=$c"
      return 0
    fi
    sleep 2
  done
  log "NO_ROOM for ${need_mib}MiB after 3600s (commit=$(commit_bytes))"
  return 1
}

# job <name> <mib> <wall> <work> <command...>
job() {
  local name=$1 mib=$2 wall=$3 work=$4
  shift 4
  local attempt=1
  while [ $attempt -le 8 ]; do
    wait_room "$mib" || return 9
    "$P" scripts/run_bounded.py --memory-mib "$mib" --wall-seconds "$wall" --work-items "$work" \
      --receipt "runs/$name-job.json" -- "$@" >> "$LOG" 2>&1
    local code=$?
    local status
    status=$("$P" -c "import json;print(json.load(open('runs/$name-job.json')).get('status'))" 2>/dev/null || echo unreadable)
    log "job $name attempt=$attempt exit=$code status=$status"
    if [ "$code" = "125" ] && [ "$status" = "memory_pressure" ]; then
      mv "runs/$name-job.json" "runs/$name-refused-$attempt-job.json"
      [ -e "runs/$name.log" ] && mv "runs/$name.log" "runs/$name-refused-$attempt.log"
      [ -e "runs/$name.stderr.log" ] && mv "runs/$name.stderr.log" "runs/$name-refused-$attempt.stderr.log"
      attempt=$((attempt+1))
      sleep 30
      continue
    fi
    return $code
  done
  log "job $name exhausted pressure retries"
  return 9
}

COMMON=(--model pilot --curriculum all-class-interleaved-v1)

log "=== sequence start ==="
job accumulation-profile-balanced-r3 640 300 16 "$P" scripts/train_m2_stream.py \
  --output runs/accumulation-profile-balanced-r3 "${COMMON[@]}" \
  --accumulation-policy balanced-eight-v1 --profile-square --steps 2 \
  > runs/accumulation-profile-balanced-r3.log 2> runs/accumulation-profile-balanced-r3.stderr.log
log "profile balanced exit=$?"

job accumulation-arm-balanced-s1 640 600 512 "$P" scripts/train_m2_stream.py \
  --output runs/accumulation-arm-balanced-s1 "${COMMON[@]}" \
  --accumulation-policy balanced-eight-v1 --steps 128 --max-new-steps 64 --wall-seconds 540 \
  --admission runs/accumulation-profile-balanced-r3 \
  > runs/accumulation-arm-balanced-s1.log 2> runs/accumulation-arm-balanced-s1.stderr.log
log "balanced segment 1 exit=$?"

job accumulation-arm-balanced-s2 640 600 512 "$P" scripts/train_m2_stream.py \
  --output runs/accumulation-arm-balanced-s2 "${COMMON[@]}" \
  --accumulation-policy balanced-eight-v1 --steps 128 --max-new-steps 64 --wall-seconds 540 \
  --resume-from runs/accumulation-arm-balanced-s1 \
  --admission runs/accumulation-profile-balanced-r3 \
  > runs/accumulation-arm-balanced-s2.log 2> runs/accumulation-arm-balanced-s2.stderr.log
log "balanced segment 2 exit=$?"

job accumulation-profile-online-r3 640 300 2 "$P" scripts/train_m2_stream.py \
  --output runs/accumulation-profile-online-r3 "${COMMON[@]}" \
  --accumulation-policy online-interleaved-v1 --profile-square --steps 2 \
  > runs/accumulation-profile-online-r3.log 2> runs/accumulation-profile-online-r3.stderr.log
log "profile online exit=$?"

job accumulation-arm-online-s1 640 600 512 "$P" scripts/train_m2_stream.py \
  --output runs/accumulation-arm-online-s1 "${COMMON[@]}" \
  --accumulation-policy online-interleaved-v1 --steps 1024 --max-new-steps 512 --wall-seconds 540 \
  --admission runs/accumulation-profile-online-r3 \
  > runs/accumulation-arm-online-s1.log 2> runs/accumulation-arm-online-s1.stderr.log
log "online segment 1 exit=$?"

job accumulation-arm-online-s2 640 600 512 "$P" scripts/train_m2_stream.py \
  --output runs/accumulation-arm-online-s2 "${COMMON[@]}" \
  --accumulation-policy online-interleaved-v1 --steps 1024 --max-new-steps 512 --wall-seconds 540 \
  --resume-from runs/accumulation-arm-online-s1 \
  --admission runs/accumulation-profile-online-r3 \
  > runs/accumulation-arm-online-s2.log 2> runs/accumulation-arm-online-s2.stderr.log
log "online segment 2 exit=$?"

log "=== sequence end ==="
for n in accumulation-profile-balanced-r3 accumulation-arm-balanced-s1 accumulation-arm-balanced-s2 \
         accumulation-profile-online-r3 accumulation-arm-online-s1 accumulation-arm-online-s2; do
  if [ -e "runs/$n-job.json" ]; then
    log "$n $("$P" -c "import json;j=json.load(open('runs/$n-job.json'));print(j.get('status'),j.get('exit_code'),j.get('elapsed_seconds'),j.get('peak_job_commit_bytes'))" 2>/dev/null)"
  else
    log "$n NO_RECEIPT"
  fi
done
