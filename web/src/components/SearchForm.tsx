import { useState } from "react";
import type { AnalyzeRequest } from "../api/types";

// Client-side mirrors of the backend validation limits (see
// src/paperdna/api/validation.py). These give instant feedback; the
// backend is still the source of truth and re-validates on submit.
const MAX_CLAIM_CHARS = 1_000;
const MAX_ABSTRACT_CHARS = 12_000;

type Mode = "claim" | "abstract";

export interface SearchFormProps {
  onSubmit: (req: AnalyzeRequest) => void;
  loading?: boolean;
}

// The landing form: a large claim input, an optional abstract toggle, and
// the primary "Analyze Research" action. Validation errors render inline.
export default function SearchForm({ onSubmit, loading }: SearchFormProps) {
  const [mode, setMode] = useState<Mode>("claim");
  const [claim, setClaim] = useState("");
  const [abstract, setAbstract] = useState("");
  const [error, setError] = useState<string | null>(null);

  const claimTooLong = claim.length > MAX_CLAIM_CHARS;
  const abstractTooLong = abstract.length > MAX_ABSTRACT_CHARS;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = claim.trim();
    if (!trimmed) {
      setError("Enter a scientific claim to analyze.");
      return;
    }
    if (claimTooLong) {
      setError(
        `Claim is too long (${claim.length} characters; the limit is ${MAX_CLAIM_CHARS}).`
      );
      return;
    }
    if (abstractTooLong) {
      setError(
        `Abstract is too long (${abstract.length} characters; the limit is ${MAX_ABSTRACT_CHARS}).`
      );
      return;
    }
    setError(null);
    onSubmit({
      claim: trimmed,
      abstract: mode === "abstract" && abstract.trim() ? abstract.trim() : null,
    });
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex gap-2" role="tablist" aria-label="Input mode">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "claim"}
          onClick={() => setMode("claim")}
          className={
            mode === "claim"
              ? "rounded-t-md border border-b-0 border-slate-300 bg-white px-4 py-1.5 text-sm font-medium text-accent"
              : "rounded-t-md border border-slate-200 bg-slate-50 px-4 py-1.5 text-sm text-slate-500"
          }
        >
          Claim
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "abstract"}
          onClick={() => setMode("abstract")}
          className={
            mode === "abstract"
              ? "rounded-t-md border border-b-0 border-slate-300 bg-white px-4 py-1.5 text-sm font-medium text-accent"
              : "rounded-t-md border border-slate-200 bg-slate-50 px-4 py-1.5 text-sm text-slate-500"
          }
        >
          Claim + abstract
        </button>
      </div>

      <div className="rounded-b-md rounded-tr-md border border-slate-300 bg-white p-4 shadow-sm">
        <label htmlFor="claim" className="sr-only">
          Scientific claim
        </label>
        <textarea
          id="claim"
          data-testid="claim-input"
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          placeholder="e.g. Retrieval augmentation improves question-answering accuracy."
          className="w-full resize-y border-0 p-0 text-lg text-ink placeholder:text-slate-400 focus:outline-none focus:ring-0"
          rows={2}
        />

        {mode === "abstract" && (
          <>
            <label htmlFor="abstract" className="sr-only">
              Optional abstract
            </label>
            <textarea
              id="abstract"
              data-testid="abstract-input"
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              placeholder="Paste an abstract to verify the claim's provenance (optional)."
              className="mt-3 w-full resize-y rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-ink placeholder:text-slate-400 focus:border-accent focus:outline-none"
              rows={5}
            />
            <p className="mt-1 text-xs text-slate-400">
              If the claim appears verbatim in the abstract, its evidence is
              verified against it.
            </p>
          </>
        )}

        {error && (
          <p role="alert" className="mt-2 text-sm text-red-600">
            {error}
          </p>
        )}

        <div className="mt-4 flex items-center justify-between">
          <span className="text-xs text-slate-400">
            {claim.length}/{MAX_CLAIM_CHARS}
          </span>
          <button
            type="submit"
            disabled={loading}
            data-testid="analyze-button"
            className="rounded-md bg-accent px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analyzing…" : "Analyze Research"}
          </button>
        </div>
      </div>
    </form>
  );
}
