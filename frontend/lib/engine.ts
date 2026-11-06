/* Jahan engine domain — TypeScript mirror of simcore/schemas.
   Vocabulary follows CONTEXT.md (brief, claim, assumption ledger, audience,
   population, scenario, world, digest, finding, trace view, ...).
   All data comes from the engine's own artefacts via /api/* — nothing here is
   mock state. */

export type ClaimSource = "user_asserted" | "public_source" | "assumed";
export type FieldOrigin = "measured" | "extracted" | "synthesized" | "calibrated";
export type PersonaFieldDomain =
  | "demographic" | "psychographic" | "category_behaviour"
  | "economic" | "decision_rule" | "media";
/* Each field domain in the words the pages use, in the order they are offered. */
export const DOMAIN_WORDS: [PersonaFieldDomain, string][] = [
  ["demographic", "Who they are"], ["psychographic", "How they think"], ["category_behaviour", "What they do"],
  ["economic", "Money & work"], ["decision_rule", "How they decide"], ["media", "What they read & watch"],
];
export type TickUnit = "hour" | "day" | "week";
export type InterventionKind = "launch" | "teaser" | "promotion";
export type Channel = "survey_room" | "social_feed" | "forum" | "wom";
export type TurnTask = "reaction" | "first_seen" | "conversation" | "reflection" | "purchase" | "claim_audit" | "probe";
export type BeliefDim = "value" | "fit" | "trust";
export type TrustLevel = "uncalibrated" | "category_benchmarked" | "prospectively_validated";
export type Confidence = "low" | "medium" | "high";
export type FindingKind = "ranking" | "risk" | "objection" | "belief_shift" | "wom_path" | "intent_trajectory" | "recommendation";
export type AnomalyKind = "herding" | "backlash" | "flop";
export type DegradationRung = "warn" | "freeze_optional_tier_b" | "subsample_activation" | "pause" | "wave_unaffordable";
export type RunStatus = "running" | "completed" | "paused" | "partial";
export type WorldStatus = "completed" | "partial" | "not_started";
export type RelaxationRung = "widen_ordinal" | "drop_filter" | "accept_shortfall";
export type GateReference = "design" | "category_targets";
export type EventKind =
  | "stimulus_published" | "exposure_dropped" | "turn" | "guardrail_violation"
  | "reflection" | "memory" | "belief_snapshot" | "probe" | "cost"
  | "intervention" | "degraded" | "tick_closed" | "lifecycle";
export type CostSource = "gateway" | "price_table" | "estimate" | "cache" | "unknown";

/* ---------- brief ---------- */
export interface Claim {
  id: string;
  text: string;
  source: ClaimSource;
  evidence_url?: string | null;
}
export interface Price { amount: number; currency: string }
export interface Competitor { name: string; price?: Price | null; claims: string[] }
export interface Audience {
  name: string;
  share?: number | null;
  /* An exact value, a list of alternatives, or `{ range: [first, last] }` over the ordinal bands. */
  attribute_filters: Record<string, string | number | (string | number)[] | { range: [string, string] }>;
}
export interface Assumption { text: string; source: ClaimSource }
export interface Brief {
  product: { name: string; category: string; description: string };
  price: Price;
  claims: Claim[];
  competitors: Competitor[];
  target_market: string;
  audiences: Audience[];
  assumptions: Assumption[];
  ontology_version: string;
}

/* ---------- category ontology ---------- */
export interface OrdinalBand { label: string; midpoint: number }
export interface CategoryOntology {
  category: string;
  version: string;
  attribute_domains: Record<string, PersonaFieldDomain>;
  conditioning_set: string[];
  relevance_order: string[];
  anchor_sets: Record<string, string>;
  completion_policy: { completable_domains: PersonaFieldDomain[] };
  ordinal_scales: { attribute: string; bands: OrdinalBand[] }[];
  targets?: {
    source: string;
    marginals: Record<string, Record<string, number>>;
  } | null;
}

/* ---------- population: gates + manifest ---------- */
export interface GateResult {
  kind: "categorical" | "ordinal" | "graph";
  attribute?: string;
  check?: string;
  chi_square?: number;
  degrees_of_freedom?: number;
  p_value?: number;
  adjusted_p_value?: number | null;
  significance_level?: number;
  ks_statistic?: number;
  ks_similarity?: number;
  similarity_threshold?: number;
  measured?: number;
  threshold?: number;
  passed: boolean;
  reference: GateReference;
}
export interface Relaxation {
  audience: string;
  rung: RelaxationRung;
  attribute?: string | null;
  authored?: unknown;
  applied?: unknown;
  rows_before: number;
  rows_after: number;
  share_achieved: number;
}
export interface GateReport {
  results: GateResult[];
  source_mix: Record<string, number>;
  attribute_origins: Record<string, FieldOrigin>;
  achieved_mix: Record<string, number>;
  relaxations: Relaxation[];
  overall: boolean;
  evidence: FieldOrigin;
  reference: GateReference;
}
export interface PopulationManifest {
  population_hash: string;
  graph_hash?: string | null;
  population_seed: number;
  persona_ids: string[];
  requested_mix: Record<string, number>;
  achieved_mix: Record<string, number>;
  synthesized_share: number;
  completion?: { model_id: string; template_id: string; template_hash: string } | null;
  parameters?: unknown;
}
export interface PersonaRecord {
  persona_id: string;
  source: string;
  conditioning: Record<string, string>;
  attributes: Record<string, string>;
  origins: Record<string, FieldOrigin>;
}

/* ---------- run ---------- */
export interface ModelPinRef { model_id: string; role: string }
export interface Variant {
  variant_id: string;
  name: string;
  description: string;
  emphasized_claims: string[];
}
export interface Scenario {
  scenario_hash?: string;
  /* A version's name (ADR 0052): presentation, outside every identity hash; a rename is served over it. */
  label?: string | null;
  variant: Variant;
  price: Price;
  audience_weights: Record<string, number>;
  tick_unit: TickUnit;
  horizon_ticks: number;
  interventions: { tick: number; kind: InterventionKind }[];
  exposure_budget: number;
  channels: Channel[];
  survey_every: number;
  launch_reach: number;
}
export interface WorldOutcome {
  world_id: string;
  status: WorldStatus;
  last_closed_tick?: number | null;
  rungs: DegradationRung[];
}
export interface WorldProgress {
  world_id: string;
  last_closed_tick?: number | null;
  turns?: number;
  rungs?: string[];
  /* Turns per channel — survey_room is the waves' own — and the ticks whose wave was answered. */
  turns_by_channel?: Record<string, number>;
  waves_answered?: number[];
}
export interface RunSummary {
  run_id: string;
  status: RunStatus;
  engine_version?: string;
  recorded_cost: number;
  discarded_ticks: number;
  config_hash?: string;
  seeds: number[];
  budget?: { max_cost: number; currency: string } | null;
  scenarios: Scenario[];
  world_ids: string[];
  outcomes: WorldOutcome[];
  has_gate_report: boolean;
  has_report: boolean;
  trust_level?: TrustLevel | null;
  finding_count?: number;
  fake?: boolean;
  live?: boolean;
  progress?: WorldProgress[];
  has_trace_summary?: boolean;
  /* Why a study that is not running and has no report stopped: the last thing it said. */
  launch_error?: string | null;
  /* What tells one run from another, read from its own files. */
  product?: string | null;
  category?: string | null;
  personas?: number | null;
  created_at?: string | null;
}

/* ---------- digest ---------- */
export type PMF5 = [number, number, number, number, number];
export interface OutcomeDigest {
  scenario_hash: string;
  tick_unit: TickUnit;
  seed: number;
  world_id: string;
  adoption?: number | null;
  unmeasured_reason?: string | null;
  polarization?: number | null;
  polarization_reason?: string | null;
  audience_divergence?: number | null;
  audience_pmfs: Record<string, number[]>;
  audience_shares: Record<string, number>;
  community_pmfs: Record<string, number[]>;
  community_sizes: Record<string, number>;
  turns_without_intent: number;
  turn_count: number;
  action_mix: Record<string, number>;
  belief_movement_mean: Record<BeliefDim, number>;
  belief_movement_abs: Record<BeliefDim, number>;
  belief_move_mean: number;
  wom_deliveries: number;
  wom_reach: number;
  rungs: DegradationRung[];
  /* Intent over time (ADR 0048): one entry per survey wave; the headline is the last wave's. */
  waves?: WaveDigest[];
  exposures_by_channel?: Record<string, number>;
  reached_by_channel?: Record<string, number>;
}
export interface WaveDigest {
  tick: number;
  respondents: number;
  audience_pmfs: Record<string, number[]>;
  audience_shares: Record<string, number>;
  audience_adoption: Record<string, number>;
  adoption?: number | null;
  reached: number;
  unreached: number;
  reached_adoption?: number | null;
  unreached_adoption?: number | null;
}
export interface ScenarioSummary {
  scenario_hash: string;
  tick_unit: TickUnit;
  adoption_spread?: number | null;
  polarization_spread?: number | null;
  divergence_spread?: number | null;
  belief_move_spread?: number | null;
  rung_mixed: boolean;
}

/* ---------- report ---------- */
export interface TrustStatement { level: TrustLevel; caveats: string[] }
export interface Finding {
  finding_id: string;
  kind: FindingKind;
  statement: string;
  evidence_trace_ids: string[];
  disconfirming_test: string;
  confidence: Confidence;
  ranked_scenarios?: string[];
}
export interface ObjectionCluster {
  label: string;
  verbatim_trace_ids: string[];
  size: number;
  threshold: number;
  embed_model_id: string;
  world_id?: string | null;
}
export interface ScenarioMethod {
  variant_id: string;
  channels: string[];
  survey_every: number;
  horizon_ticks: number;
  launch_reach?: number | null;
}
export interface MethodDisclosure {
  scenarios?: ScenarioMethod[];
  departures?: string[];
  pins: ModelPinRef[];
  fallbacks: ModelPinRef[];
  seeds: number[];
  template_hashes: { name: string; hash: string }[];
  anchor_set_hashes: { name: string; hash: string }[];
}
export interface ClusteringCoverage {
  world_id: string;
  category: "objection" | "reason";
  eligible: number;
  clustered: number;
  excluded: number;
}
export interface ClusteringExclusion {
  world_id: string;
  category: "objection" | "reason";
  text: string;
  trace_id: string;
  reason_code: string;
  reason: string;
}
export interface StudyReport {
  contract_version: string;
  run_id: string;
  config_hash: string;
  engine_commit: string;
  trust: TrustStatement;
  findings: Finding[];
  objection_clusters: ObjectionCluster[];
  /* What persuaded, grouped as objections are (ADR 0053). Absent on reports written before it. */
  reason_clusters?: ObjectionCluster[];
  clustering_coverage?: ClusteringCoverage[] | null;
  clustering_exclusions?: ClusteringExclusion[];
  digests: OutcomeDigest[];
  assumptions: Assumption[];
  method: MethodDisclosure;
  validation: string;
  forced_from?: string[];
  forced_inputs?: string[];
}

/* ---------- trace view (the fixed question set) ---------- */
export interface TraceEvent {
  event_id: string;
  world_id: string;
  tick: number;
  seq: number;
  persona_id?: string | null;
  payload: { kind: EventKind; [k: string]: unknown };
}
export interface BeliefPoint { tick: number; beliefs: Record<string, number> }
export interface TraceEdge { u: string; v: string; channel: string; count: number; last_tick: number }
export interface VerbatimRecord {
  event_id: string;
  persona_id: string;
  tick: number;
  text: string;
  action: string;
}
export interface VerbatimGroup { key: string; count: number; samples: VerbatimRecord[] }
export interface UITrace {
  run_id: string;
  worlds: string[];
  event_counts: Record<string, Record<string, number>>;
  max_tick: Record<string, number>;
  belief_histories: Record<string, Record<string, BeliefPoint[]>>;
  belief_personas: Record<string, string[]>;
  edges_top: TraceEdge[];
  verbatim_groups: Record<string, VerbatimGroup[]>;
  costs: { role: string; calls: number; input_tokens: number; output_tokens: number; cost: number | null }[];
  recorded_cost: number;
  resolved: Record<string, TraceEvent>;
}

/* ---------- misc ---------- */
export interface OntologyRef { category: string; version: string; path: string }

export const EVIDENCE_ORDER: FieldOrigin[] = ["measured", "calibrated", "extracted", "synthesized"];

export function top2box(pmf: number[]): number {
  return (pmf[3] ?? 0) + (pmf[4] ?? 0);
}
export function pmfMean(pmf: number[]): number {
  return pmf.reduce((s, v, i) => s + v * (i + 1), 0);
}

/* What Who you study saves and a study starts from: audiences with the one ontology version they were drafted against. */
export interface AudienceSet {
  id: string; name: string; category: string; ontology_version: string; created_at: string; description: string;
  audiences: { name: string; share: number | null; attribute_filters: Record<string, string | string[]> }[];
  assumptions: { text: string; source: string }[]; sources: string[]; study_size: number;
}
