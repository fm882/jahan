"""A resumed run never re-runs — or re-closes — a world that reached its horizon.

Re-emitting the closing event appends a duplicate behind a sink that keeps everything
and crashes against a record that is finalized and read-only, so completed cells are
skipped where their last closed tick is read.
"""

import pytest

from simcore.runner import run
from simcore.schemas import BriefPack, EventFilter, Population, RunConfig, canonical_hash
from simcore.trace import TraceStore, UnknownWorldError
from simcore.trace.finalize import read_finalized_events
from tests.boundary.runner.test_tick_loop import FakeWorld, _completed
from tests.study_builders import pack_payload, population_payload, run_config_payload, scenario_payload


def _pack():
    return BriefPack.model_validate(pack_payload())


def _population():
    return Population.model_validate(population_payload())


def _config(population, pack, horizon=3, seeds=(4021,)):
    scenario = scenario_payload(horizon_ticks=horizon, interventions=[])
    payload = run_config_payload(scenarios=[scenario], seeds=list(seeds))
    payload.update({
        "brief_hash": canonical_hash(pack.brief),
        "ontology_hash": canonical_hash(pack.ontology),
        "population_hash": population.manifest.population_hash,
        "graph_hash": population.manifest.graph_hash,
    })
    return RunConfig.model_validate(payload)


class _ResumableStore(TraceStore):
    """The on-disk store behind the runner's trace seam: live worlds answer from their
    live record, finalized ones from their lasting one."""

    def __init__(self, root, run_id):
        super().__init__(root)
        self._run_id = run_id

    def events_for(self, world_id):
        if self.is_finalized(self._run_id, world_id):
            return read_finalized_events(self._root, self._run_id, world_id)
        try:
            return self.read_live(self._run_id, world_id)
        except UnknownWorldError:
            return ()

    def all_events(self):
        out = []
        for world_id in self.world_ids(self._run_id):
            out.extend(self.events_for(world_id))
        return tuple(out)


def _agent(jobs):
    return tuple(
        _completed(job, job.presentation.impression.tick * 100 + index) for index, job in enumerate(jobs)
    )


def test_rerunning_a_completed_world_writes_nothing_and_completes(tmp_path):
    pack, population = _pack(), _population()
    config = _config(population, pack)
    run_id = config.run_id

    def factory(header):
        store.create_world(run_id, header)
        return FakeWorld(header)

    store = _ResumableStore(tmp_path, run_id)
    first = run(config, pack=pack, population=population, trace=store, registry=store.registry,
                world_factory=factory, agent_fn=_agent)
    assert first.status.value == "completed"
    before = list(store.view(run_id, first.outcomes[0].world_id).events(EventFilter()))

    second = run(config, pack=pack, population=population, trace=store, registry=store.registry,
                 world_factory=factory, agent_fn=_agent)
    assert second.status.value == "completed"
    assert second.outcomes == first.outcomes
    after = list(store.view(run_id, first.outcomes[0].world_id).events(EventFilter()))
    assert [event.event_id for event in after] == [event.event_id for event in before]


def test_a_run_is_not_completed_while_a_later_world_is_still_running(tmp_path):
    """Worlds run one after another and each is finalized as it ends. The first world's lasting record
    must not make the run read as complete: everything that reads the registry while the second world
    runs (the overview, the run status) would say it had finished."""
    pack, population = _pack(), _population()
    config = _config(population, pack, seeds=(4021, 917731))
    run_id = config.run_id
    store = _ResumableStore(tmp_path, run_id)

    def factory(header):
        store.create_world(run_id, header)
        return FakeWorld(header)

    seen: list[tuple[str, str]] = []
    first_world: list[str] = []

    def progress(world_id, tick, spent):
        first_world.append(world_id) if not first_world else None
        if world_id != first_world[0]:
            seen.append((world_id, store.registry.entry(run_id).status.value))

    result = run(config, pack=pack, population=population, trace=store, registry=store.registry,
                 world_factory=factory, agent_fn=_agent, progress=progress)

    assert seen, "the second world reported no progress, so nothing was observed"
    assert {status for _, status in seen} == {"running"}
    assert result.status.value == "completed"
    assert store.registry.entry(run_id).status.value == "completed"


def test_resume_of_mixed_worlds_skips_finished_world_and_saved_ticks(tmp_path):
    pack, population = _pack(), _population()
    config = _config(population, pack, seeds=(4021, 917731))
    store = _ResumableStore(tmp_path, config.run_id)
    started = []

    def factory(header):
        started.append(header.world_id)
        store.create_world(config.run_id, header)
        return FakeWorld(header)

    def interrupt(world_id, tick, spent):
        if len(started) == 2 and tick == 0:
            raise RuntimeError("interrupt after a saved tick in the second world")

    with pytest.raises(RuntimeError, match="interrupt after"):
        run(config, pack=pack, population=population, trace=store, registry=store.registry,
            world_factory=factory, agent_fn=_agent, progress=interrupt)
    first, second = started
    assert store.is_finalized(config.run_id, first)
    before = tuple(store.events_for(first))
    partial = tuple(store.events_for(second))
    started.clear()
    ticks = []

    def agent(jobs):
        ticks.extend(job.presentation.impression.tick for job in jobs)
        return _agent(jobs)

    resumed = run(config, pack=pack, population=population, trace=store, registry=store.registry,
                  world_factory=factory, agent_fn=agent)
    assert resumed.status.value == "completed"
    assert started == [second] and ticks and min(ticks) == 1
    assert tuple(store.events_for(first)) == before
    assert tuple(store.events_for(second))[:len(partial)] == partial
