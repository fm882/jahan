"""The interface, exercised through the real proxy against the real engine API.

The surface tests elsewhere in this directory read the interface's source for strings, which
cannot fail when the interface is broken — the list of ontologies changed shape at the proxy
and the intake page could not load, and every one of them still passed. This one starts the
engine API and the Next.js server, and asks the interface's own routes for what its pages ask.

It needs the frontend's dependencies (`npm install` in `frontend/`) and `node`; without them
it is skipped, and says so. The engine API runs in-process on a free port; nothing here reaches
a network beyond loopback.
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FRONTEND = REPO / "frontend"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not (FRONTEND / "node_modules" / "next").is_dir(),
    reason="the interface's dependencies are not installed: `npm install` in frontend/",
)


# The suite refuses every outgoing connection so that nothing reaches a network. This test talks to
# two servers it started itself, so it lifts that for loopback and for nothing else.
_connect = socket.socket.connect
_create_connection = socket.create_connection
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _loopback_only(sock, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in _LOOPBACK:
        raise RuntimeError(f"tests must not open network connections: {host}")
    return _connect(sock, address)


@pytest.fixture(autouse=True)
def _loopback_is_allowed(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _loopback_only)
    monkeypatch.setattr(socket, "create_connection", _create_connection)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(url: str, timeout: float = 120.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as refusal:
        body = refusal.read()
        try:
            return refusal.code, json.loads(body)
        except ValueError:
            return refusal.code, body.decode("utf-8", "replace")


def _post(url: str, body: dict, timeout: float = 120.0):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as refusal:
        return refusal.code, json.loads(refusal.read() or b"null")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """A fake study and a gate-only run on disk, the engine API over them, the interface over that."""
    import uvicorn

    from simcore.web import create_app
    from tests.boundary.cli.support import fake_args, run_command

    work = tmp_path_factory.mktemp("interface")
    runs = work / "runs"
    finished = "run-" + "0" * 24 + "91"
    cwd = os.getcwd()
    os.chdir(work)
    try:
        code, output = run_command(*fake_args(runs, finished))
    finally:
        os.chdir(cwd)
    assert code == 0, output

    # A three-attribute corpus and a copy of the shipped ontologies: the builder saves for real,
    # and what it saves must never land in the repository.
    corpus = work / "corpus"
    corpus.mkdir()
    (corpus / "persona_codes.schema.json").write_text(json.dumps({"columns": [
        {"id": "age_bracket", "values": ["18-24", "25-34", "35-44"]},
        {"id": "exercise_frequency", "values": ["never", "weekly", "daily"]},
        {"id": "protein_habit", "values": ["low", "high"]},
    ]}))
    ontologies = work / "ontologies"
    shutil.copytree(REPO / "ontologies", ontologies)

    api_port = _free_port()
    app = create_app(
        runs_dir=runs, ontology_dir=ontologies,
        anchors_dir=REPO / "anchors", engine_root=REPO, corpus_dir=corpus,
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=api_port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.1)
    assert server.started, "the engine API did not start"
    api = f"http://127.0.0.1:{api_port}"

    brief = (REPO / "examples" / "protein_water.yaml").read_text()
    evidence = json.loads((REPO / "examples" / "protein_water.yaml.evidence.json").read_text())
    status, gated = _post(f"{api}/api/gate", {"brief_yaml": brief, "evidence_json": evidence, "n": 60})
    assert status == 200 and gated["run_id"], gated

    web_port = _free_port()
    dist = ".next-test"  # named in tsconfig's include, so Next has nothing to rewrite there
    env = {
        **os.environ,
        "SIMCORE_WEB_URL": api,
        "NEXT_DIST_DIR": dist,
        "NEXT_TELEMETRY_DISABLED": "1",
    }
    log = (work / "next.log").open("wb")
    web = subprocess.Popen(
        [str(FRONTEND / "node_modules" / ".bin" / "next"), "dev", "-p", str(web_port)],
        cwd=FRONTEND, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{web_port}"
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                _get(f"{base}/api/status", timeout=30)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(0.5)
        else:
            raise AssertionError("the interface did not start:\n" + (work / "next.log").read_text()[-2000:])
        yield {
            "base": base, "api": api, "finished": finished, "gated": gated["run_id"], "runs": runs,
            "ontologies": ontologies,
        }
    finally:
        web.terminate()
        try:
            web.wait(timeout=15)
        except subprocess.TimeoutExpired:
            web.kill()
        log.close()
        shutil.rmtree(FRONTEND / dist, ignore_errors=True)
        server.should_exit = True
        thread.join(timeout=10)


def test_the_first_screen_reads_the_workspace_the_engine_derived(stack):
    status, body = _get(f"{stack['base']}/api/overview")
    assert status == 200
    assert body["workspace"]["total_studies"] == 1 and body["workspace"]["reports_written"] == 1
    assert {run["run_id"] for run in body["runs"]} == {stack["finished"], stack["gated"]}
    assert body["ontologies"] and body["briefs"]


def test_lists_arrive_the_way_the_pages_read_them(stack):
    """`/api/ontologies` came back as `{ontologies: [...]}` and the intake page took it for a list."""
    status, ontologies = _get(f"{stack['base']}/api/ontologies")
    assert status == 200 and isinstance(ontologies, list) and ontologies
    first = ontologies[0]
    status, whole = _get(f"{stack['base']}/api/ontologies?category={first['category']}&version={first['version']}")
    assert status == 200 and whole["category"] == first["category"]

    status, runs = _get(f"{stack['base']}/api/runs")
    assert status == 200 and isinstance(runs, list) and len(runs) == 2

    status, briefs = _get(f"{stack['base']}/api/briefs")
    assert status == 200 and isinstance(briefs, list)
    protein = next(brief for brief in briefs if brief["name"] == "protein_water")
    assert protein["brief"]["competitors"], "the brief's competitors survive the trip"


def test_a_finished_study_arrives_with_its_evidence_resolved(stack):
    status, detail = _get(f"{stack['base']}/api/runs/{stack['finished']}")
    assert status == 200
    assert detail["summary"]["status"] == "completed" and detail["summary"]["fake"] is True
    assert detail["report"]["trust"]["level"] == "uncalibrated"
    assert detail["trace"]["worlds"]
    cited = {
        trace_id
        for finding in detail["report"]["findings"]
        for trace_id in finding.get("evidence_trace_ids", [])[:12]
    }
    assert cited <= set(detail["trace"]["resolved"]), "every cited event resolves to the event itself"
    # The fake personas only praise, so since ADR 0053 the evidence they leave is a reason to buy, not an objection.
    assert detail["report"]["reason_clusters"] and not detail["report"]["objection_clusters"]


def test_a_failed_or_gate_only_run_is_readable(stack):
    """The population page's whole reason to exist: a study that never ran. It answered 500."""
    status, detail = _get(f"{stack['base']}/api/runs/{stack['gated']}")
    assert status == 200, detail
    assert detail["gate"]["results"] and detail["manifest"]["persona_ids"]
    assert detail["trace"] is None and detail["report"] is None
    assert detail["summary"]["has_gate_report"] is True


def test_an_unknown_run_is_a_404_with_a_reason(stack):
    status, body = _get(f"{stack['base']}/api/runs/run-nope")
    assert status == 404 and "no record" in body["error"]


def test_a_refusal_keeps_the_reason_and_the_status_the_engine_gave(stack):
    """Every refusal came back as a bare 502 whose message was the status code."""
    status, body = _post(f"{stack['base']}/api/briefs/validate", {"brief_yaml": "product: {}"})
    assert status == 422 and "brief.yaml: product.name" in body["error"], body

    status, body = _post(f"{stack['base']}/api/runs", {"brief_yaml": "x: 1", "n": 0})
    assert status == 422 and "n" in body["error"], body

    status, body = _post(f"{stack['base']}/api/runs", {"brief_yaml": "x: 1", "api_key": "sk-secret"})
    assert status == 422 and "api_key" in body["error"] and "sk-secret" not in json.dumps(body)


def test_the_engine_being_away_is_said_plainly(stack):
    """The proxy reports what it could not reach, not a stack trace."""
    env_missing = _get(f"{stack['base']}/api/status")
    assert env_missing[0] == 200 and env_missing[1]["fake_available"] is True


def test_the_pages_render_against_the_engine(stack):
    for path in (
        "/", "/intake", "/who", "/calibration", f"/population?run={stack['gated']}",
        f"/population?run={stack['finished']}", f"/run?run={stack['finished']}",
        f"/trace?run={stack['finished']}", f"/report?run={stack['finished']}",
        f"/atlas?run={stack['finished']}", f"/calibration?run={stack['finished']}",
    ):
        with urllib.request.urlopen(f"{stack['base']}{path}", timeout=180) as response:
            assert response.status == 200, path
            html = response.read().decode("utf-8", "replace")
        assert "Application error" not in html and "Internal Server Error" not in html, path


def test_the_trust_page_states_the_engines_own_floors(stack):
    status, trust = _get(f"{stack['api']}/api/trust")
    assert status == 200 and trust["levels"][0] == "uncalibrated"
    with urllib.request.urlopen(f"{stack['base']}/calibration", timeout=180) as response:
        html = response.read().decode("utf-8", "replace")
    assert f"{trust['floors']['distribution_similarity']:.2f}" in html


def _ontology(**over) -> dict:
    base = {
        "category": "snacks", "version": "1.0.0",
        "attribute_domains": {"age_bracket": "demographic", "protein_habit": "psychographic"},
        "conditioning_set": ["age_bracket"],
        "relevance_order": ["age_bracket", "protein_habit"],
        "completion_policy": {"completable_domains": []},
        "ordinal_scales": [{"attribute": "age_bracket", "bands": [
            {"label": "18-24", "midpoint": 21.0}, {"label": "25-34", "midpoint": 29.5}]}],
        "anchor_sets": {"purchase_intent": "purchase-intent-v1"},
    }
    return {**base, **over}


def test_the_builders_refusals_and_saves_reach_the_browser_intact(stack):
    """The reason a name is refused, and that a saved version is never overwritten — through the
    interface's own routes. Both were lost at the proxy: a refusal read `502`, and the reason went with it."""
    base = stack["base"]

    typo = json.loads(json.dumps(_ontology()).replace("protein_habit", "protein_habbit"))
    status, body = _post(f"{base}/api/ontologies/validate", {"ontology": typo})
    assert status == 422
    assert "protein_habbit" in body["error"] and "protein_habit" in body["error"], "names what resembles it"

    status, body = _post(f"{base}/api/ontologies", {"ontology": _ontology()})
    assert status == 201 and body == {"category": "snacks", "version": "1.0.0"}
    saved = stack["ontologies"] / "snacks" / "1.0.0.json"
    before = saved.read_bytes()

    status, body = _post(f"{base}/api/ontologies", {"ontology": _ontology()})
    assert status == 409 and "already exists" in body["error"]
    status, _ = _post(f"{base}/api/ontologies", {"ontology": _ontology(version="1.1.0")})
    assert status == 201
    assert saved.read_bytes() == before, "an existing version is left exactly as it was"

    status, listed = _get(f"{base}/api/ontologies")
    assert {(o["category"], o["version"]) for o in listed} >= {("snacks", "1.0.0"), ("snacks", "1.1.0")}
    assert not (REPO / "ontologies" / "snacks").exists(), "nothing was written into the repository"


def test_the_codebook_search_is_the_corpus_own(stack):
    status, body = _get(f"{stack['base']}/api/codebook?query=age")
    assert status == 200 and body["attributes"][0]["id"] == "age_bracket"
    assert body["attributes"][0]["values"] == ["18-24", "25-34", "35-44"]


def _through_the_form(brief: dict) -> str:
    """A brief as the intake form holds it and writes it out — the frontend's own code, run by node."""
    script = (
        "import { briefToYaml, formFromBrief } from %r;"
        "let raw = ''; for await (const chunk of process.stdin) raw += chunk;"
        "process.stdout.write(briefToYaml(formFromBrief(JSON.parse(raw))));"
    ) % (FRONTEND / "lib" / "briefYaml.ts").as_uri()
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(brief), capture_output=True, text=True, cwd=FRONTEND, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_every_brief_the_engine_holds_survives_the_intake_form(stack):
    """The form's YAML emitter flattened a nested audience filter into a sibling key and broke on a
    description with a line break, so a brief loaded into the form and validated came back refused —
    and it dropped the competitors and rewrote every assumption as asserted."""
    status, briefs = _get(f"{stack['base']}/api/briefs")
    assert status == 200 and briefs
    checked = 0
    for entry in briefs:
        status, loaded = _get(f"{stack['base']}/api/briefs/{entry['name']}")
        assert status == 200
        yaml_text = _through_the_form(loaded["brief"])
        status, ledger = _post(
            f"{stack['base']}/api/briefs/validate",
            {"brief_yaml": yaml_text, "evidence_json": loaded["evidence"] or {}},
        )
        assert status == 200, (entry["name"], ledger, yaml_text)
        assert ledger["audiences"] == [audience["name"] for audience in loaded["brief"]["audiences"]]
        assert ledger["claims"] == [claim["id"] for claim in loaded["brief"]["claims"]]
        checked += 1
    assert checked >= 2


def test_a_form_with_an_awkward_description_is_still_a_brief_the_engine_reads(stack):
    status, loaded = _get(f"{stack['base']}/api/briefs/protein_water")
    brief = loaded["brief"]
    brief["product"]["description"] = "Kellogg's \"protein\" water:\nclear, 20g - no sugar # really"
    brief["audiences"][0]["attribute_filters"] = {"exercise_frequency": "3_plus_weekly"}
    yaml_text = _through_the_form(brief)
    status, ledger = _post(
        f"{stack['base']}/api/briefs/validate", {"brief_yaml": yaml_text, "evidence_json": loaded["evidence"] or {}}
    )
    assert status == 200, (ledger, yaml_text)


# --- the pages, rendered by a browser ------------------------------------------------
#
# Everything above asks the interface's routes. A page is code that runs in a browser, and a page that
# throws while it renders answers 200 to every request the routes can be asked: the run page crashed on
# every study — its pin table has roles pinned to nothing — and no test above could have said so.

_CHROME = next(
    (found for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
     if (found := shutil.which(name))),
    None,
)
needs_a_browser = pytest.mark.skipif(_CHROME is None, reason="no Chrome or Chromium to render pages with")


def _render(base: str, path: str) -> tuple[str, str]:
    """The page after a browser has run it, and what the browser logged. Non-loopback hosts do not resolve."""
    done = subprocess.run(
        [
            _CHROME, "--headless=new", "--no-sandbox", "--disable-gpu",
            "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1",
            "--enable-logging=stderr", "--v=0", "--virtual-time-budget=30000",
            "--dump-dom", f"{base}{path}",
        ],
        capture_output=True, text=True, timeout=240,
    )
    return done.stdout, done.stderr


def _visible(html: str) -> str:
    import html as _html
    import re

    text = re.sub(r"<style.*?</style>|<script.*?</script>", " ", html, flags=re.S)
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)))


@needs_a_browser
def test_report_discloses_unavailable_clustering_and_reads_legacy_coverage(stack):
    root = stack["runs"] / stack["finished"]
    path = root / "report.json"
    original = path.read_text()
    report = json.loads(original)
    world = report["digests"][0]["world_id"]
    from simcore.schemas import EventFilter
    from simcore.trace import TraceStore

    events = TraceStore(root / "trace").view(stack["finished"], world).events(EventFilter(kinds=("turn",)))
    event = next(item for item in events if item.payload.turn.reaction.verbatim)
    report["objection_clusters"] = []
    report["clustering_coverage"] = [dict(world_id=world, category="objection", eligible=1, clustered=0, excluded=1)]
    report["clustering_exclusions"] = [dict(
        world_id=world, category="objection", text=event.payload.turn.reaction.verbatim,
        trace_id=event.event_id, reason_code="non_finite", reason="The embedding remained non-finite after one fresh retry.",
    )]
    try:
        path.write_text(json.dumps(report))
        html, log = _render(stack["base"], f"/report?run={stack['finished']}")
        text = _visible(html)
        assert "Clustering unavailable. 1 of 1 objections could not be grouped." in text
        assert "No objections were found" not in text
        assert event.payload.turn.reaction.verbatim in text
        assert f"resolve={event.event_id}" in html
        assert "Application error" not in html and "Uncaught" not in log
        legacy = json.loads(original)
        legacy.pop("clustering_coverage", None)
        legacy.pop("clustering_exclusions", None)
        path.write_text(json.dumps(legacy))
        text = _visible(_render(stack["base"], f"/report?run={stack['finished']}")[0])
        assert "Clustering coverage was not recorded in this report." in text
    finally:
        path.write_text(original)


@needs_a_browser
@pytest.mark.parametrize(
    "page",
    ["/", "/intake", "/who", "/calibration", "/run", "/report", "/atlas", "/trace", "/population"],
)
def test_no_page_throws_while_it_renders(stack, page):
    for run in (stack["finished"], stack["gated"]):
        path = f"{page}?run={run}" if page not in ("/", "/intake", "/who") else page
        html, log = _render(stack["base"], path)
        thrown = [line for line in log.splitlines() if "Uncaught" in line]
        assert "Application error" not in html and not thrown, (path, thrown[:2])


@needs_a_browser
def test_a_failed_or_gate_only_population_shows_its_gates(stack):
    html, _ = _render(stack["base"], f"/population?run={stack['gated']}")
    text = _visible(html)
    assert "Distribution gates" in text and "Passes" in text, "each gate shows its statistic and threshold"
    assert "Audience mix" in text


@needs_a_browser
def test_the_run_page_of_a_fake_study_says_so_and_names_its_pins(stack):
    text = _visible(_render(stack["base"], f"/run?run={stack['finished']}")[0])
    assert "fake study" in text
    assert "Chat A fake/tier-a-1" in text and "Chat B fake/tier-b-1" in text, "pinned roles are listed by what they do"
    assert "Safety" not in text, "roles pinned to nothing are not"


@needs_a_browser
def test_the_shell_shows_the_engines_own_state_and_nothing_invented(stack):
    text = _visible(_render(stack["base"], "/")[0])
    for invented in ("Team plan", "$412", "1,500", "Oral-care", "Refill pouch", "v0.4.1"):
        assert invented not in text, invented
    assert "Recent studies" in text and stack["finished"][:16] in text
    assert "spent" in text


@needs_a_browser
def test_a_brief_loaded_into_intake_keeps_its_competitors_and_filters(stack):
    """The form dropped the brief's competitors on load, so running the example ran a different study."""
    text = _visible(_render(stack["base"], "/intake")[0])
    assert "Clear whey shooter" in text, "the brief's competitor is in the YAML the engine will read"
    assert "exercise_frequency" in text and "3_plus_weekly" in text
