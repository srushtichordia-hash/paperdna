import { Link } from "react-router-dom";
import type { DiscoveryResponse } from "../api/types";
import { PaperList } from "../components/PaperList";
import { RelationCard } from "../components/RelationCard";
import KnowledgeGraph from "../components/KnowledgeGraph";
import { EmptyState } from "../components/StateStates";

function CapabilitiesBanner({ result }: { result: DiscoveryResponse }) {
  const { capabilities, rankingKind } = result;
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span>
          Ranking:{" "}
          <strong data-testid="ranking-kind" className="text-ink">
            {rankingKind === "semantic" ? "semantic similarity" : "lexical overlap"}
          </strong>
          {rankingKind === "lexical" && (
            <span className="ml-1 text-xs text-amber-700">
              (fallback — not meaning-aware)
            </span>
          )}
        </span>
        {capabilities.embedderName && (
          <span className="text-xs text-slate-400">
            embedder: {capabilities.embedderName}
          </span>
        )}
        {capabilities.nliModel && (
          <span className="text-xs text-slate-400">
            NLI: {capabilities.nliModel}
          </span>
        )}
        {!capabilities.nliModel && (
          <span className="text-xs text-slate-400">
            NLI model not loaded — no relationship predictions
          </span>
        )}
      </div>
      {result.messages.length > 0 && (
        <ul className="mt-2 list-inside list-disc space-y-0.5 text-xs text-slate-500">
          {result.messages.map((m, i) => (
            <li key={i}>{m}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function ResultsPage({
  result,
}: {
  result: DiscoveryResponse | null;
}) {
  if (!result) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-16">
        <EmptyState message="No analysis to show. Run a search first." />
        <div className="mt-4 text-center">
          <Link to="/" className="text-accent hover:underline">
            ← Back to search
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex min-h-full max-w-4xl flex-col px-6 py-10">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-ink">
          Paper<span className="text-accent">DNA</span>
        </h1>
        <Link to="/" className="text-sm text-accent hover:underline">
          ← New search
        </Link>
      </header>

      <p className="mt-4 rounded-lg bg-white p-4 text-sm text-slate-700 shadow-sm">
        <span className="font-medium">Your claim:</span> {result.claim}
      </p>

      <div className="mt-4">
        <CapabilitiesBanner result={result} />
      </div>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-ink">Ranked papers</h2>
        <div className="mt-3">
          <PaperList papers={result.papers} />
        </div>
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-ink">
          Predicted relationships
        </h2>
        {result.relations.length === 0 ? (
          <div className="mt-3">
            <EmptyState message="No relationships were predicted for this claim." />
          </div>
        ) : (
          <ul className="mt-3 space-y-3">
            {result.relations.map((rel, i) => (
              <RelationCard key={i} relation={rel} />
            ))}
          </ul>
        )}
      </section>

      <section className="mt-8">
        <h2 className="text-lg font-semibold text-ink">Knowledge graph</h2>
        <p className="mt-1 text-sm text-slate-500">
          Click a node for details. Dashed edges are model predictions, not
          verified facts.
        </p>
        <div className="mt-3">
          <KnowledgeGraph nodes={result.graph.nodes} edges={result.graph.edges} />
        </div>
      </section>

      <footer className="mt-12 border-t border-slate-200 pt-4 text-xs text-slate-400">
        PaperDNA predictions are model-generated hypotheses, not verified
        scientific facts. Always consult the cited sources.
      </footer>
    </div>
  );
}
