import type { RelationOut } from "../api/types";
import { EvidenceQuote } from "./EvidenceQuote";

const LABEL_META: Record<string, { label: string; className: string }> = {
  entailment: { label: "supports", className: "bg-green-100 text-green-800" },
  contradiction: {
    label: "contradicts",
    className: "bg-red-100 text-red-800",
  },
  neutral: { label: "neutral", className: "bg-slate-100 text-slate-700" },
};

function conditionStatusColor(status: string): string {
  switch (status) {
    case "matching":
      return "text-green-700";
    case "differing":
      return "text-red-700";
    case "uncertain":
      return "text-amber-700";
    default:
      return "text-slate-500";
  }
}

export function RelationCard({ relation }: { relation: RelationOut }) {
  const interp = relation.interpretation;
  const meta = interp ? LABEL_META[interp.label] : null;

  return (
    <li
      data-testid="relation-card"
      className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {meta && (
            <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${meta.className}`}>
              {meta.label}
            </span>
          )}
          <span className="text-xs text-slate-400">
            {relation.kind === "user_vs_paper"
              ? "your claim vs. paper"
              : "paper vs. paper"}
          </span>
        </div>
        {relation.interpretation?.uncertain && (
          <span
            data-testid="uncertain-badge"
            className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800"
          >
            uncertain
          </span>
        )}
      </div>

      {interp && (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-slate-600">
          <span>
            confidence <strong className="text-ink">{interp.confidence.toFixed(2)}</strong>
          </span>
          <span className="text-xs text-slate-400">
            model: {interp.model}
          </span>
        </div>
      )}

      {!interp && (
        <p className="mt-2 text-sm text-slate-500">
          No relationship prediction (NLI model not loaded).
        </p>
      )}

      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <EvidenceQuote evidence={relation.premise} />
        <EvidenceQuote evidence={relation.hypothesis} />
      </div>

      {relation.conditions && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <p className="text-xs font-medium text-slate-500">
            Experimental conditions
          </p>
          <ul className="mt-1 space-y-0.5 text-sm">
            {relation.conditions.fields
              .filter((f) => f.status !== "missing")
              .map((f) => (
                <li key={f.category} className="flex justify-between gap-2">
                  <span className="text-slate-500">{f.category}</span>
                  <span className={`font-medium ${conditionStatusColor(f.status)}`}>
                    {f.status}
                    {f.premiseValues.length > 0 && (
                      <span className="ml-1 text-slate-500">
                        ({f.premiseValues.join(", ")}
                        {f.hypothesisValues.length > 0 &&
                          ` vs. ${f.hypothesisValues.join(", ")}`}
                        )
                      </span>
                    )}
                  </span>
                </li>
              ))}
          </ul>
        </div>
      )}

      {interp && (
        <p className="mt-3 text-xs text-slate-400">
          Model prediction, not a verified scientific fact.{" "}
          {!interp.scoresAreCalibrated &&
            "NLI scores are uncalibrated; treat confidence as a heuristic."}
        </p>
      )}
    </li>
  );
}
