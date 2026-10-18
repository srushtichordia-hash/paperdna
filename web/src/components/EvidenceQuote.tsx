import type { EvidenceOut, VerificationStatus } from "../api/types";

// Human-readable label + color for each verification status. This is the
// provenance surface: it is always shown, never hidden, so a quote is never
// presented as more (or less) verified than it actually is.
const VERIFICATION_META: Record<
  VerificationStatus,
  { label: string; className: string }
> = {
  verified: {
    label: "verified against abstract",
    className: "bg-green-100 text-green-800",
  },
  verified_by_search: {
    label: "verified by search",
    className: "bg-green-100 text-green-800",
  },
  unverified: {
    label: "unverified",
    className: "bg-amber-100 text-amber-800",
  },
  unavailable: {
    label: "evidence unavailable",
    className: "bg-slate-100 text-slate-600",
  },
};

export function VerificationBadge({ status }: { status: VerificationStatus }) {
  const meta = VERIFICATION_META[status];
  return (
    <span
      data-testid={`verification-${status}`}
      className={`rounded-full px-2 py-0.5 text-xs font-medium ${meta.className}`}
    >
      {meta.label}
    </span>
  );
}

export function EvidenceQuote({ evidence }: { evidence: EvidenceOut }) {
  const offsetsKnown = evidence.claim.start != null && evidence.claim.end != null;
  return (
    <figure className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-slate-500">
          {evidence.paper.title ?? `${evidence.paper.source}:${evidence.paper.sourceId}`}
        </span>
        <VerificationBadge status={evidence.verification} />
      </div>
      {evidence.verification === "unavailable" ? (
        <p className="mt-2 text-sm italic text-slate-500">
          {evidence.unavailableReason ?? "No evidence text available."}
        </p>
      ) : (
        <blockquote className="mt-2 border-l-2 border-slate-300 pl-3 text-sm text-ink">
          “{evidence.sentence}”
        </blockquote>
      )}
      <figcaption className="mt-2 text-xs text-slate-400">
        {offsetsKnown
          ? `chars ${evidence.claim.start}–${evidence.claim.end}`
          : "offsets unknown"}
        {evidence.unavailableReason && evidence.verification !== "unavailable"
          ? ` · ${evidence.unavailableReason}`
          : ""}
      </figcaption>
    </figure>
  );
}
