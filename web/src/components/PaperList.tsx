import type { PaperOut } from "../api/types";

// Human label for the ranking kind. Lexical MUST NOT be presented as
// semantic similarity — the wording is deliberately explicit.
function rankingBadge(paper: PaperOut) {
  if (paper.rankingKind === "semantic") {
    return (
      <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
        semantic similarity
      </span>
    );
  }
  if (paper.rankingKind === "lexical") {
    return (
      <span
        data-testid="lexical-badge"
        className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800"
      >
        lexical overlap (fallback)
      </span>
    );
  }
  return null;
}

export function PaperCard({ paper, rank }: { paper: PaperOut; rank: number }) {
  return (
    <li className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <span className="text-xs font-semibold text-slate-400">
            #{rank}
          </span>
          <h3 className="mt-1 font-medium text-ink">
            {paper.title ?? `${paper.source}:${paper.sourceId}`}
          </h3>
          {paper.authors.length > 0 && (
            <p className="mt-0.5 text-sm text-slate-500">
              {paper.authors.join(", ")}
              {paper.published ? ` · ${paper.published}` : ""}
            </p>
          )}
        </div>
        <div className="flex flex-col items-end gap-1">
          {paper.score != null && (
            <span className="text-sm font-semibold text-ink">
              {paper.score.toFixed(3)}
            </span>
          )}
          {rankingBadge(paper)}
        </div>
      </div>
      {(paper.url || paper.doi) && (
        <div className="mt-2 flex gap-3 text-sm">
          {paper.url && (
            <a
              href={paper.url}
              target="_blank"
              rel="noreferrer"
              className="text-accent hover:underline"
            >
              Source
            </a>
          )}
          {paper.doi && (
            <span className="text-slate-500">DOI: {paper.doi}</span>
          )}
        </div>
      )}
    </li>
  );
}

export function PaperList({ papers }: { papers: PaperOut[] }) {
  if (papers.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        No paper in the corpus is related to this claim.
      </p>
    );
  }
  return (
    <ol className="space-y-3">
      {papers.map((paper, i) => (
        <PaperCard key={`${paper.source}:${paper.sourceId}`} paper={paper} rank={i + 1} />
      ))}
    </ol>
  );
}
