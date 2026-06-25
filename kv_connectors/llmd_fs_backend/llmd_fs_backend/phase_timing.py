# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-phase timing for the vLLM OffloadingConnector (trace mode only).

The OffloadingConnector is a thin facade that delegates every phase of a
request to an inner scheduler-side object (engine-core process) or worker-side
object (GPU worker process):

    scheduler  get_num_new_matched_tokens -> "lookup"   (is the block offloaded?)
               update_state_after_alloc   -> "alloc"    (reserve blocks to load)
               build_connector_meta       -> "meta"     (what to load/store now)
               request_finished           -> "finished"
    worker     start_kv_transfers         -> "read"     (GET / load from storage)
               prepare_store_kv           -> "write"    (PUT / store to storage)
               get_finished               -> "poll"     (reap completed transfers)

This module monkey-patches those inner methods to accumulate wall-clock per
phase, then flushes a `connector_phase:` line once per engine step (on the
last method of the step: build_connector_meta for the scheduler, get_finished
for the worker). Each line carries the DELTA since the previous flush, so a
test harness can SUM the lines over a request and the parts add up to the
total time that request spent inside the connector:

    connector_phase: role=worker read=1.234:1 write=5.678:1 poll=0.012:1 total=6.924

(`<phase>=<milliseconds>:<calls>`.) Patching the inner classes — rather than
the OffloadingConnector facade — lets us separate read/write/poll, which the
facade lumps together inside its single get_finished() override.

Active only when STORAGE_LOG_LEVEL is TRACE/DEBUG; otherwise install is a
no-op and there is zero per-call overhead.
"""

from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger(__name__)

PATCHED_FLAG = "llmd_fs_phase_timed"

_TRACE_ENABLED = os.environ.get("STORAGE_LOG_LEVEL", "INFO").upper() in (
    "TRACE",
    "DEBUG",
)

# role -> {phase_name: [cumulative_time_s_since_flush, call_count_since_flush]}
_acc: dict[str, dict[str, list]] = {"sched": {}, "worker": {}}
_lock = threading.Lock()


def _record(role: str, phase: str, dt: float) -> None:
    with _lock:
        slot = _acc[role].setdefault(phase, [0.0, 0])
        slot[0] += dt
        slot[1] += 1


def _flush(role: str) -> None:
    """Emit and reset the accumulated per-phase deltas for *role*."""
    with _lock:
        phases = _acc[role]
        if not phases:
            return
        # Preserve a stable, meaningful column order per role.
        order = (
            ["lookup", "alloc", "meta", "finished"]
            if role == "sched"
            else ["read", "write", "poll"]
        )
        names = [p for p in order if p in phases] + [
            p for p in phases if p not in order
        ]
        parts = " ".join(
            f"{p}={phases[p][0] * 1000:.3f}:{phases[p][1]}" for p in names
        )
        total_ms = sum(t for t, _ in phases.values()) * 1000
        phases.clear()
    log.debug("connector_phase: role=%s %s total=%.3f", role, parts, total_ms)


def install_connector_phase_timing_patch() -> None:
    """Patch the inner scheduler/worker connector classes to time each phase.

    Idempotent. No-op unless STORAGE_LOG_LEVEL is a trace level.
    """
    if not _TRACE_ENABLED:
        return
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
            OffloadingConnectorScheduler as Sched,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
            OffloadingConnectorWorker as Worker,
        )
    except Exception as exc:  # vLLM layout changed / connector unavailable
        log.debug("skipping connector phase timing patch: %s", exc)
        return

    # Force this child logger to DEBUG so the connector_phase line reaches the
    # handler installed on the package logger in trace mode.
    log.setLevel(logging.DEBUG)

    def patch(cls, method_name: str, role: str, phase: str, flush: bool = False):
        orig = getattr(cls, method_name, None)
        if orig is None:
            log.debug("phase timing: %s.%s missing, skipping", cls.__name__, method_name)
            return
        if getattr(orig, PATCHED_FLAG, False):
            return

        def timed(self, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(self, *args, **kwargs)
            finally:
                _record(role, phase, time.perf_counter() - t0)
                # `flush` methods are the last connector call of an engine step,
                # so flushing here emits one delta line per step for this role.
                if flush:
                    _flush(role)

        setattr(timed, PATCHED_FLAG, True)
        setattr(cls, method_name, timed)

    # Scheduler-side phases (engine-core process). build_connector_meta runs
    # every scheduler step and is the natural flush point.
    patch(Sched, "get_num_new_matched_tokens", "sched", "lookup")
    patch(Sched, "update_state_after_alloc", "sched", "alloc")
    patch(Sched, "request_finished", "sched", "finished")
    patch(Sched, "build_connector_meta", "sched", "meta", flush=True)

    # Worker-side phases (GPU worker process). get_finished runs every step
    # (after read+write are queued) and is the natural flush point.
    patch(Worker, "start_kv_transfers", "worker", "read")
    patch(Worker, "prepare_store_kv", "worker", "write")
    patch(Worker, "get_finished", "worker", "poll", flush=True)

    log.info("installed connector phase timing patch")
