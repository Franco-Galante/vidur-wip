from math import ceil
from typing import List, Optional

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class FifoReplicaScheduler(BaseReplicaScheduler):
    """Simple FIFO replica scheduler with memory-based admission and preemption.

    Admission rule: a new request is admitted only when the KV cache occupancy
    after its prefill allocation would not exceed m_star * num_blocks (and the
    number of concurrent requests is below batch_size_cap).

    Preemption: when a decode-phase request needs a new KV block but none are
    free, a victim is selected (youngest or oldest in-flight decode request)
    and restarted — its tokens are freed and it is re-queued for prefill.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_running_batches: int = 0
        self._preempted_requests: List[Request] = []
        # Maximum blocks that may be occupied when a new request is admitted.
        self._m_star_blocks: int = int(self._config.m_star * self._config.num_blocks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _can_admit_new_request(self, request: Request) -> bool:
        """Return True if the request passes the memory-only admission gate."""
        if len(self._allocation_map) >= self._config.batch_size_cap:
            return False
        num_required_blocks = ceil(
            request.num_prefill_tokens / self._config.block_size
        )
        return (
            self._num_allocated_blocks + num_required_blocks <= self._m_star_blocks
        )

    def _can_allocate_decode(self) -> bool:
        """Return True if at least one KV block is free."""
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _needs_new_block(self, request: Request) -> bool:
        """Return True if the request has consumed all its reserved KV blocks."""
        num_tokens_reserved = (
            self._allocation_map[request.id] * self._config.block_size
        )
        return request.num_processed_tokens > num_tokens_reserved

    def _preempt_victim(
        self, preempted_copy: List[Request]
    ) -> bool:
        """Preempt one victim from preempted_copy per preemption_policy.

        Returns True if a victim was preempted, False if the list was empty.
        """
        if not preempted_copy:
            return False
        if self._config.preemption_policy == "oldest":
            victim = preempted_copy.pop(0)
        elif self._config.preemption_policy == "youngest":
            victim = preempted_copy.pop(-1)
        else:
            raise ValueError(f"Invalid preemption policy: {self._config.preemption_policy}")
        victim.restart()
        self.free(victim.id)
        self._request_queue.insert(0, victim)
        return True

    # ------------------------------------------------------------------
    # Scheduler interface
    # ------------------------------------------------------------------

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1
        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_next_batch(self) -> Optional[Batch]:
        requests: List[Request] = []
        num_tokens: List[int] = []

        # --- Phase 1: schedule in-flight decode requests (FIFO order) ---
        # Work on a local copy so preemption removals don't corrupt iteration.
        self._preempted_requests.sort(key=lambda r: r.arrived_at) # sort to work on arrival order and not batch completion
        preempted_copy = list(self._preempted_requests)
        self._preempted_requests = []

        while preempted_copy:
            request = preempted_copy.pop(0)
            assert request.is_prefill_complete

            if self._needs_new_block(request):
                # Try to free space by preempting victims.
                while not self._can_allocate_decode():
                    if not self._preempt_victim(preempted_copy):
                        # No victims remain; preempt the current request itself.
                        request.restart()
                        self.free(request.id)
                        self._request_queue.insert(0, request)
                        request = None
                        break

                if request is None:
                    continue

                self.allocate(request.id, 1)

            requests.append(request)
            num_tokens.append(1)

        # --- Phase 2: admit new requests from the FIFO queue ---
        num_batch_tokens = sum(num_tokens)
        while self._request_queue:
            next_request = self._request_queue[0]
            if num_batch_tokens + next_request.num_prefill_tokens > self._config.max_tokens_in_batch:
                break
            if not self._can_admit_new_request(next_request):
                # Head-of-line blocking: preserve strict FIFO.
                break
            request = self._request_queue.pop(0)
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            requests.append(request)
            num_tokens.append(request.num_prefill_tokens)
            num_batch_tokens += request.num_prefill_tokens

        if not requests:
            return None

        return Batch(self._replica_id, requests, num_tokens)
