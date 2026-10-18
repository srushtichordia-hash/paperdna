import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvidenceQuote, VerificationBadge } from "../components/EvidenceQuote";
import { RelationCard } from "../components/RelationCard";
import type { EvidenceOut, RelationOut } from "../api/types";

function makeEvidence(overrides: Partial<EvidenceOut> = {}): EvidenceOut {
  return {
    paper: {
      source: "demo",
      sourceId: "p001",
      title: "A Paper",
      url: null,
      doi: null,
      published: null,
    },
    claim: {
      claimId: "demo:p001:0-10",
      source: "demo",
      sourceId: "p001",
      start: 0,
      end: 10,
    },
    sentence: "Retrieval helps.",
    verification: "verified",
    unavailableReason: null,
    ...overrides,
  };
}

function makeRelation(overrides: Partial<RelationOut> = {}): RelationOut {
  return {
    kind: "user_vs_paper",
    premise: makeEvidence(),
    hypothesis: makeEvidence({
      claim: {
        claimId: "user:query:unknown",
        source: "user",
        sourceId: "query",
        start: null,
        end: null,
      },
      verification: "unverified",
      unavailableReason: null,
    }),
    interpretation: {
      label: "entailment",
      scores: { entailment: 0.7, contradiction: 0.1, neutral: 0.2 },
      confidence: 0.7,
      uncertain: false,
      model: "fake-nli",
      conditionSummary: null,
      conditionStats: null,
      notes: [],
      scoresAreCalibrated: false,
    },
    conditions: null,
    ...overrides,
  };
}

describe("VerificationBadge", () => {
  it("renders each verification status distinctly", () => {
    render(<VerificationBadge status="verified" />);
    expect(screen.getByTestId("verification-verified")).toBeInTheDocument();
  });

  it("marks unverified evidence clearly", () => {
    render(<VerificationBadge status="unverified" />);
    expect(screen.getByTestId("verification-unverified")).toBeInTheDocument();
  });
});

describe("EvidenceQuote", () => {
  it("shows the sentence and offset range when known", () => {
    render(<EvidenceQuote evidence={makeEvidence()} />);
    expect(screen.getByText(/Retrieval helps\./)).toBeInTheDocument();
    expect(screen.getByText(/chars 0–10/)).toBeInTheDocument();
  });

  it("shows unknown offsets for a text-only user claim", () => {
    render(
      <EvidenceQuote
        evidence={makeEvidence({
          claim: {
            claimId: "user:query:unknown",
            source: "user",
            sourceId: "query",
            start: null,
            end: null,
          },
          verification: "unverified",
        })}
      />
    );
    expect(screen.getByText(/offsets unknown/)).toBeInTheDocument();
  });

  it("shows the reason for unavailable evidence instead of a quote", () => {
    render(
      <EvidenceQuote
        evidence={makeEvidence({
          verification: "unavailable",
          unavailableReason: "No abstract available.",
        })}
      />
    );
    expect(screen.getByText(/No abstract available\./)).toBeInTheDocument();
  });
});

describe("RelationCard", () => {
  it("labels the prediction and shows uncertainty badge when uncertain", () => {
    render(
      <RelationCard
        relation={makeRelation({
          interpretation: {
            label: "neutral",
            scores: {},
            confidence: 0.4,
            uncertain: true,
            model: "fake-nli",
            conditionSummary: null,
            conditionStats: null,
            notes: [],
            scoresAreCalibrated: false,
          },
        })}
      />
    );
    expect(screen.getByTestId("uncertain-badge")).toBeInTheDocument();
  });

  it("states that predictions are not verified facts", () => {
    render(<RelationCard relation={makeRelation()} />);
    expect(
      screen.getByText(/model prediction, not a verified scientific fact/i)
    ).toBeInTheDocument();
  });

  it("warns that NLI scores are uncalibrated", () => {
    render(<RelationCard relation={makeRelation()} />);
    expect(screen.getByText(/uncalibrated/i)).toBeInTheDocument();
  });

  it("shows both evidence sides with verification badges", () => {
    render(<RelationCard relation={makeRelation()} />);
    expect(screen.getByTestId("verification-verified")).toBeInTheDocument();
    expect(screen.getByTestId("verification-unverified")).toBeInTheDocument();
  });

  it("explains when no prediction is available", () => {
    render(<RelationCard relation={makeRelation({ interpretation: null })} />);
    expect(
      screen.getByText(/no relationship prediction/i)
    ).toBeInTheDocument();
  });
});
