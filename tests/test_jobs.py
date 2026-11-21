"""Tests for ai_diffusion/model/jobs.py - history pruning of the job queue."""

from __future__ import annotations

import pytest

from ai_diffusion.image import Bounds, Extent, Image, ImageCollection
from ai_diffusion.model.jobs import Job, JobKind, JobParams, JobQueue, protected_job_ids
from ai_diffusion.settings import settings


def _finish(queue: JobQueue, id: str, accepted: tuple[int, ...] = (), size=256):
    """Add a finished diffusion job with 2 result images (~0.5 MB for size 256)."""
    job = queue.add_job(Job(id, JobKind.diffusion, JobParams(Bounds(0, 0, size, size), id)))
    for index in accepted:
        job.in_use[index] = True
    images = [Image.create(Extent(size, size)) for _ in range(2)]
    queue.set_results(job, ImageCollection(images))
    queue.notify_finished(job)
    return job


def test_protected_job_ids(monkeypatch: pytest.MonkeyPatch):
    jobs = [("a", {0: True}), ("b", {}), ("c", {0: True, 1: True}), (None, {0: True})]

    monkeypatch.setattr(settings, "history_accepted_limit", 0)
    assert protected_job_ids(jobs) == {"a", "c"}

    monkeypatch.setattr(settings, "history_accepted_limit", 2)
    assert protected_job_ids(jobs) == {"c"}  # newest 2 accepted images

    monkeypatch.setattr(settings, "history_accepted_limit", 3)
    assert protected_job_ids(jobs) == {"a", "c"}


def test_prune_keeps_accepted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "history_size", 1)  # MB
    monkeypatch.setattr(settings, "history_accepted_limit", 200)

    queue = JobQueue()
    for i in range(6):
        _finish(queue, f"job{i}", accepted=(1,) if i == 1 else ())
    assert [j.id for j in queue] == ["job1", "job5"]

    queue = JobQueue()
    for i in range(6):
        _finish(queue, f"job{i}")
    assert [j.id for j in queue] == ["job4", "job5"]


def test_prune_accepted_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "history_size", 1)  # MB
    monkeypatch.setattr(settings, "history_accepted_limit", 2)

    queue = JobQueue()
    accepted = {0: (0,), 1: (0, 1), 2: (1,)}
    for i in range(6):
        _finish(queue, f"job{i}", accepted=accepted.get(i, ()))
    # job2 (1 image) fits the limit, job1 (2 images) would exceed it -> no longer protected
    assert [j.id for j in queue] == ["job2", "job5"]


def test_notify_unused():
    queue = JobQueue()
    job = _finish(queue, "job", size=8)
    assert not job.is_used
    queue.notify_used("job", 1)
    assert job.is_used and job.result_was_used(1)
    queue.notify_unused("job", 1)
    assert not job.is_used


def test_clear_keep_used():
    queue = JobQueue()
    for i in range(3):
        _finish(queue, f"job{i}", size=8)
    queue.notify_used("job1", 0)

    queue.clear(keep_used=True)
    assert [j.id for j in queue] == ["job1"]

    queue.clear()
    assert len(queue) == 0
