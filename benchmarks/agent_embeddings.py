"""Offline baseline/current comparison through the real embedding client and runner.

Run with the repository's Python environment. --source selects an exported revision.
Only an in-process HTTP mock is used; no credentials or external inference are read.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", required=True, help="revision of the selected source tree")
    parser.add_argument("--personas", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--delay-ms", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 20 <= args.personas <= 50 or args.ticks < 3 or args.repeats < 2 or args.delay_ms < 0:
        parser.error("use 20–50 personas, at least three ticks, two repetitions and nonnegative delay")
    sys.path.insert(0, str(args.source.resolve()))

    import httpx
    from simcore.agent import AgentConfig, advance_state, turns
    from simcore.inference import ExecutionSettings, InferenceClient
    from simcore.ports.fake import FakeChat, FakeEmbed
    from simcore.runner import InMemoryRegistry, InMemoryTraceSink, run
    from simcore.schemas import (
        BriefPack, CompletedTurn, MemoryEvent, ModelPins, Population,
        RunConfig, TracePartition, canonical_hash,
    )
    from simcore.world import World, WorldConfig
    from tests.boundary.agent.support import answering, by_template, make_job, memory_dict, reflecting
    from tests.study_builders import persona_payload, population_payload, run_config_payload, scenario_payload, stimulus_id, with_derived_hashes

    model = "benchmark/embed"
    pins = ModelPins.model_validate({"tier_a": "fake/chat", "tier_b": "fake/chat", "embed": model})
    settings = {
        "base_url": "http://offline-benchmark.test/v1", "max_concurrency": 16,
        "embeddings_batch_size": 64, "max_retries": 0,
        "requests_per_minute": 1000000, "tokens_per_minute": 100000000,
    }

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def execute(kind, cache_dir):
        metrics = {"port_invocations": 0, "submitted_texts": 0, "upstream_requests": 0,
                   "upstream_tokens": 0, "billed_cost": 0.0, "allocated_cost": 0.0,
                   "preparation_seconds": 0.0, "chat_seconds": 0.0, "finalization_seconds": 0.0}

        async def handler(request):
            body = json.loads(request.content)
            await asyncio.sleep(args.delay_ms / 1000)
            texts = body["input"]
            tokens = sum(max(1, len(text) // 4) for text in texts)
            cost = tokens * 0.000001
            metrics["upstream_requests"] += 1
            metrics["upstream_tokens"] += tokens
            metrics["billed_cost"] += cost
            vectors = FakeEmbed(dim=4)
            return httpx.Response(200, json={
                "model": model, "data": [{"index": i, "embedding": vectors._vector(text)} for i, text in enumerate(texts)],
                "usage": {"prompt_tokens": tokens, "cost": cost},
            })

        client = InferenceClient(pins, ExecutionSettings(**settings, cache_dir=cache_dir), transport=httpx.MockTransport(handler))

        class Embed:
            model_id = model

            def embed(self, texts):
                metrics["port_invocations"] += 1
                metrics["submitted_texts"] += len(texts)
                return client.embed(texts)

        class Chat(FakeChat):
            first = None
            last = None

            def complete(self, requests, on_outcome=None):
                started = perf_counter()
                if self.first is None:
                    self.first = started
                result = super().complete(requests, on_outcome=on_outcome)
                self.last = perf_counter()
                metrics["chat_seconds"] += self.last - started
                return result

        chat = Chat(by_template({"persona_reflection": reflecting()}))
        config = AgentConfig(run_seed=4021, horizon_ticks=args.ticks, reflection_interval=1 if kind in {"reflections", "simulation"} else 100,
                             reflection_jitter=0, probe_share=0)
        content = []

        def agent(jobs, plan=None, **kwargs):
            chat.first = chat.last = None
            started = perf_counter()
            outcomes = turns(jobs, chat=chat, embed=Embed(), config=config, **kwargs)
            ended = perf_counter()
            metrics["preparation_seconds"] += (chat.first or ended) - started
            metrics["finalization_seconds"] += ended - (chat.last or ended)
            content.extend(outcome.model_dump(mode="json", exclude={"costs"}) for outcome in outcomes)
            metrics["allocated_cost"] += sum(cost.cost or 0 for outcome in outcomes for cost in outcome.costs)
            assert all(isinstance(outcome, CompletedTurn) for outcome in outcomes), outcomes
            return outcomes

        def job_at(index, tick, unique):
            # The baseline's fixture builder supports four archetypes; ownership is unique.
            job = make_job(index % 4, tick=tick, n=tick * args.personas + index,
                           shown=[(index + 1 if unique else 1, "interest", 0.8)])
            pid = f"p-benchmark-{index:04d}"
            return job.model_copy(update={
                "persona": job.persona.model_copy(update={"persona_id": pid}),
                "state": job.state.model_copy(update={"persona_id": pid}),
                "presentation": job.presentation.model_copy(update={
                    "impression": job.presentation.impression.model_copy(update={"persona_id": pid}),
                }),
            })

        if kind == "simulation":
            raw = population_payload()
            raw["personas"] = [persona_payload(i % 4, persona_id=f"p-benchmark-{i:04d}") for i in range(args.personas)]
            ids = [persona["persona_id"] for persona in raw["personas"]]
            raw["manifest"].update(persona_ids=ids, population_hash="00" * 32)
            raw["manifest"].pop("graph_hash", None)
            raw["graph"] = {"edges": [{"u": ids[i], "v": ids[i+1], "weight": 0.5} for i in range(len(ids)-1)]}
            raw["communities"] = [{"community_id": "benchmark", "member_ids": ids}]
            population = Population.model_validate(with_derived_hashes(raw))
            pack = population.pack
            scenario = scenario_payload(horizon_ticks=args.ticks, interventions=[])
            run_config = RunConfig.model_validate(run_config_payload(
                scenarios=[scenario], seeds=[4021], pins=pins.model_dump(mode="json"),
                brief_hash=canonical_hash(pack.brief), ontology_hash=canonical_hash(pack.ontology),
                population_hash=population.manifest.population_hash, graph_hash=population.manifest.graph_hash,
                template_hashes={"persona_turn": "ab" * 32},
            ))
            trace, registry = InMemoryTraceSink(), InMemoryRegistry()
            started = perf_counter()
            result = run(run_config, pack=pack, population=population, trace=trace, registry=registry,
                         world_factory=lambda header: World(header, population=population, config=WorldConfig()),
                         agent_fn=agent)
            metrics["simulation_seconds"] = perf_counter() - started
            assert result.status.value == "completed", result
            world = result.outcomes[0].world_id
            events = trace.events_for(world)
            # Validate after timing, separately from simulation and report generation.
            TracePartition.model_validate({"header": {
                "contract_version": "1.1.0", "config": run_config.model_dump(mode="json"),
                "pack": pack.model_dump(mode="json"), "population": population.manifest.model_dump(mode="json"),
                "scenario": scenario, "replicate_seed": 4021,
            }, "events": [event.model_dump(mode="json") for event in events]})
            state_hash = digest([event.payload.model_dump(mode="json") for event in events
                                 if event.payload.kind in {"memory", "belief_snapshot"}])
            metrics["trace_events"] = len(events)
        else:
            states = {}
            started = perf_counter()
            for tick in range(1, args.ticks + 1):
                jobs, texts = [], {}
                for index in range(args.personas):
                    job = job_at(index, tick, kind == "unique")
                    state = states.get(index, job.state)
                    if tick == 1 and kind != "empty":
                        memories = tuple(MemoryEvent.model_validate(memory_dict(index*10+n, 0, f"private {index} item {n}",
                                                                                 embedding=FakeEmbed(dim=4)._vector(f"private {index} item {n}"))) for n in range(4))
                        state = state.model_copy(update={"memories": memories})
                    jobs.append(job.model_copy(update={"state": state}))
                    number = index + 1 if kind == "unique" else 1
                    texts[stimulus_id(number)] = f"product announcement {number}"
                outcomes = agent(jobs, stimulus_texts=texts)
                for index, (job, outcome) in enumerate(zip(jobs, outcomes, strict=True)):
                    states[index] = advance_state(job.state, outcome, tick, memory_cap=config.memory_cap)
            metrics["simulation_seconds"] = perf_counter() - started
            state_hash = digest([state.model_dump(mode="json") for state in states.values()])
        metrics.update(cache_hits=client.stats.cache_hits, content_hash=digest(content),
                       prompts_hash=digest(sorted(chat.calls)), state_hash=state_hash)
        assert abs(metrics["allocated_cost"] - metrics["billed_cost"]) < 1e-9
        asyncio.run(client.aclose())
        return metrics

    results = {}
    for kind in ("empty", "repeated", "unique", "reflections", "simulation"):
        results[kind] = {}
        for cache in ("cold", "warm"):
            samples = []
            with tempfile.TemporaryDirectory(prefix="simcore-embedding-benchmark-") as root:
                directory = Path(root) / "warm"
                if cache == "warm":
                    execute(kind, directory)
                for repeat in range(args.repeats):
                    samples.append(execute(kind, directory if cache == "warm" else Path(root) / str(repeat)))
            results[kind][cache] = {
                "samples": samples,
                "median_seconds": statistics.median(sample["simulation_seconds"] for sample in samples),
                "min_seconds": min(sample["simulation_seconds"] for sample in samples),
                "max_seconds": max(sample["simulation_seconds"] for sample in samples),
            }
    output = {
        "inference": "offline HTTP mock; deterministic vectors and fixed chat responses; no network",
        "source": str(args.source.resolve()), "personas": args.personas, "ticks": args.ticks,
        "revision": args.revision, "python_version": sys.version,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_pins": pins.model_dump(mode="json"),
        "repeats": args.repeats, "mock_request_delay_ms": args.delay_ms, "settings": settings,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved offline benchmark: {args.output}")


if __name__ == "__main__":
    main()
