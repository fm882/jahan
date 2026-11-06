import test from "node:test";
import assert from "node:assert/strict";
import { analysisCostLabel, clusteringForWorld, clusteringNotice } from "../lib/clustering.ts";

const coverage = { world_id: "world", category: "objection" as const, eligible: 42, clustered: 39, excluded: 3 };

test("unreported analysis prices remain distinct from free calls", () => {
  assert.equal(analysisCostLabel(null, "primary"), "cost unreported (primary)");
  assert.equal(analysisCostLabel(0, "cache"), "0 USD (cache)");
});

test("partial coverage discloses missing response counts", () => {
  assert.equal(clusteringNotice([coverage], "objections"), "3 of 42 objections could not be grouped.");
});
test("no valid vectors means unavailable, rather than no objections", () => {
  assert.match(clusteringNotice([{ ...coverage, clustered: 0, excluded: 42 }], "objections"), /^Clustering unavailable\./);
});
test("old reports have unknown coverage and empty categories are not failures", () => {
  assert.match(clusteringNotice(undefined, "objections"), /not recorded/);
  assert.doesNotMatch(clusteringNotice([{ ...coverage, eligible: 0, clustered: 0, excluded: 0 }], "objections"), /unavailable/);
});
test("world selection keeps coverage and excluded evidence separate by category", () => {
  const rows = [coverage, { ...coverage, world_id: "another", excluded: 42, clustered: 0 }, { ...coverage, category: "reason" as const }];
  assert.deepEqual(clusteringForWorld(rows, "world", "objection"), [coverage]);
  assert.deepEqual(clusteringForWorld(rows, "another", "reason"), []);
  assert.equal(clusteringForWorld(undefined, "world", "objection"), undefined);
  assert.equal(clusteringForWorld(null, "world", "objection"), undefined);
});
