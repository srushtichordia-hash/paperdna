import { useState } from "react";

export interface ModelOptInModalProps {
  open: boolean;
  embeddingMB?: number | null;
  nliMB?: number | null;
  loading?: boolean;
  error?: string | null;
  // User explicitly confirmed — caller triggers the actual load.
  onConfirm: () => void;
  onDecline: () => void;
}

// The opt-in gate. Nothing is downloaded until the user clicks "Download &
// load". Shown when the user requests semantic features but models are not
// yet loaded.
export default function ModelOptInModal({
  open,
  embeddingMB,
  nliMB,
  loading,
  error,
  onConfirm,
  onDecline,
}: ModelOptInModalProps) {
  const [acknowledged, setAcknowledged] = useState(false);

  if (!open) return null;

  const embedSize = embeddingMB != null ? `~${embeddingMB} MB` : "large";
  const nliSize = nliMB != null ? `~${nliMB} MB` : "several hundred MB";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-optin-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
    >
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <h2 id="model-optin-title" className="text-lg font-semibold text-ink">
          Load ML models?
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          Semantic search and relationship prediction require downloading
          machine-learning models. Nothing has been downloaded yet, and nothing
          will be unless you confirm.
        </p>
        <ul className="mt-3 space-y-1 text-sm text-slate-600">
          <li>• Embedding model (semantic search): {embedSize}</li>
          <li>• NLI model (relationship prediction): {nliSize}</li>
        </ul>
        <p className="mt-3 text-sm text-slate-500">
          Without them, PaperDNA uses a lexical-overlap fallback and skips
          relationship prediction — still fully usable offline.
        </p>

        <label className="mt-4 flex items-start gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(e) => setAcknowledged(e.target.checked)}
            className="mt-0.5"
          />
          I understand these models will be downloaded.
        </label>

        {error && (
          <p role="alert" className="mt-3 text-sm text-red-600">
            {error}
          </p>
        )}

        <div className="mt-5 flex justify-end gap-3">
          <button
            onClick={onDecline}
            className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
          >
            Continue without models
          </button>
          <button
            onClick={onConfirm}
            disabled={!acknowledged || loading}
            data-testid="confirm-model-load"
            className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Loading…" : "Download & load"}
          </button>
        </div>
      </div>
    </div>
  );
}
