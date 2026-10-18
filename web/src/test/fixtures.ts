import type { DiscoveryResponse } from "../api/types";

// A representative backend response for component tests. Shaped exactly like
// the FastAPI contract in src/paperdna/api/schemas.py.
export function makeDiscoveryResponse(
  overrides: Partial<DiscoveryResponse> = {}
): DiscoveryResponse {
  return {
    claim: "Retrieval augmentation improves QA accuracy.",
    abstractProvided: false,
    rankingKind: "lexical",
    papers: [
      {
        source: "demo",
        sourceId: "p001",
        title: "Retrieval Augmented Question Answering",
        abstract: "We find retrieval improves accuracy.",
        authors: ["A. Author"],
        published: "2023",
        url: "https://example.org/p001",
        doi: null,
        score: 0.85,
        rankingKind: "lexical",
      },
    ],
    userClaim: null,
    userClaimHasRealOffsets: false,
    paperClaims: [],
    claimConditions: [],
    relations: [],
    graph: { nodes: [], edges: [] },
    capabilities: {
      rankingKind: "lexical",
      embedderName: null,
      nliModel: null,
      messages: ["Semantic retrieval is off; ranked by lexical overlap."],
    },
    messages: ["Relationship classification is off."],
    stats: { loaded: 10, unique: 10, duplicates: 0, rejected: 0 },
    ...overrides,
  };
}
