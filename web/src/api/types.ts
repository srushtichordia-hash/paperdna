// TypeScript mirror of the FastAPI response contract in
// src/paperdna/api/schemas.py. Field names are camelCase to match the
// serialized JSON exactly. Keep this file in sync with the Python models.

export type RankingKind = "semantic" | "lexical";
export type RelationLabel = "entailment" | "contradiction" | "neutral";
export type RelationKind = "user_vs_paper" | "paper_vs_paper";
export type VerificationStatus =
  | "verified"
  | "verified_by_search"
  | "unverified"
  | "unavailable";
export type ConditionStatus = "matching" | "differing" | "uncertain" | "missing";
export type GraphNodeType = "paper" | "claim" | "evidence" | "condition";
export type GraphEdgeType =
  | "claim_paper"
  | "claim_evidence"
  | "claim_condition"
  | "claim_relation";

export interface AnalyzeRequest {
  claim: string;
  abstract?: string | null;
  topK?: number;
  maxPairs?: number;
  comparePapers?: boolean;
}

export interface PaperOut {
  source: string;
  sourceId: string;
  title: string | null;
  abstract: string | null;
  authors: string[];
  published: string | null;
  url: string | null;
  doi: string | null;
  score: number | null;
  rankingKind: RankingKind | null;
}

export interface ClaimOut {
  source: string;
  sourceId: string;
  sentence: string;
  start: number | null;
  end: number | null;
  cue: string | null;
  hedges: string[];
}

export interface PaperRefOut {
  source: string;
  sourceId: string;
  title: string | null;
  url: string | null;
  doi: string | null;
  published: string | null;
}

export interface ClaimRefOut {
  claimId: string;
  source: string;
  sourceId: string;
  start: number | null;
  end: number | null;
}

export interface EvidenceOut {
  paper: PaperRefOut;
  claim: ClaimRefOut;
  sentence: string;
  verification: VerificationStatus;
  unavailableReason: string | null;
}

export interface ConditionMentionOut {
  category: string;
  value: string;
  start: number;
  end: number;
  ambiguous: boolean;
  source: string;
  sourceId: string;
}

export interface ClaimConditionsOut {
  source: string;
  sourceId: string;
  mentions: ConditionMentionOut[];
}

export interface FieldComparisonOut {
  category: string;
  status: ConditionStatus;
  premiseValues: string[];
  hypothesisValues: string[];
}

export interface ConditionComparisonOut {
  premiseSource: string;
  premiseSourceId: string;
  hypothesisSource: string;
  hypothesisSourceId: string;
  fields: FieldComparisonOut[];
  stats: Record<string, number>;
}

export interface InterpretationOut {
  label: RelationLabel;
  scores: Record<string, number>;
  confidence: number;
  uncertain: boolean;
  model: string;
  conditionSummary: string | null;
  conditionStats: Record<string, number> | null;
  notes: string[];
  scoresAreCalibrated: boolean;
}

export interface RelationOut {
  kind: RelationKind;
  premise: EvidenceOut;
  hypothesis: EvidenceOut;
  interpretation: InterpretationOut | null;
  conditions: ConditionComparisonOut | null;
}

// Node data payloads, discriminated by `kind`.
export interface PaperNodeData {
  kind: "paper";
  source: string;
  sourceId: string;
  title: string | null;
  url: string | null;
  doi: string | null;
  published: string | null;
  authors: string[];
}

export interface ClaimNodeData {
  kind: "claim";
  source: string;
  sourceId: string;
  sentence: string;
  start: number | null;
  end: number | null;
  cue: string | null;
  hedges: string[] | null;
}

export interface EvidenceNodeData {
  kind: "evidence";
  claimId: string;
  sentence: string;
  verification: VerificationStatus;
  unavailableReason: string | null;
  start: number | null;
  end: number | null;
}

export interface ConditionNodeData {
  kind: "condition";
  claimId: string;
  category: string;
  value: string;
  start: number;
  end: number;
  ambiguous: boolean;
  globalStart: number | null;
  globalEnd: number | null;
}

export type GraphNodeData =
  | PaperNodeData
  | ClaimNodeData
  | EvidenceNodeData
  | ConditionNodeData;

export interface GraphNodeOut {
  id: string;
  type: GraphNodeType;
  position: { x: number; y: number };
  data: GraphNodeData;
}

export interface GraphEdgeOut {
  id: string;
  source: string;
  target: string;
  type: GraphEdgeType;
  label: string | null;
  data: Record<string, unknown>;
}

export interface GraphOut {
  nodes: GraphNodeOut[];
  edges: GraphEdgeOut[];
}

export interface CapabilitiesOut {
  rankingKind: RankingKind;
  embedderName: string | null;
  nliModel: string | null;
  messages: string[];
}

export interface DiscoveryResponse {
  claim: string;
  abstractProvided: boolean;
  rankingKind: RankingKind;
  papers: PaperOut[];
  userClaim: ClaimOut | null;
  userClaimHasRealOffsets: boolean;
  paperClaims: ClaimOut[];
  claimConditions: Array<{ claim: ClaimOut; conditions: ClaimConditionsOut }>;
  relations: RelationOut[];
  graph: GraphOut;
  capabilities: CapabilitiesOut;
  messages: string[];
  stats: Record<string, number>;
}

export interface CorpusStatsOut {
  loaded: number;
  unique: number;
  duplicates: number;
  rejected: number;
}

export interface CorpusResponse {
  stats: CorpusStatsOut;
  papers: PaperOut[];
}

export interface ModelStatusOut {
  installed: boolean;
  loaded: boolean;
  name: string | null;
  downloadMB: number | null;
}

export interface ModelsStatusOut {
  embedding: ModelStatusOut;
  nli: ModelStatusOut;
}

export interface HealthResponse {
  status: string;
  corpus: CorpusStatsOut;
  models: ModelsStatusOut;
}
