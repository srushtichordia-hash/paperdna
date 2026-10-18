import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import SearchForm from "../components/SearchForm";
import ModelOptInModal from "../components/ModelOptInModal";
import { ErrorState } from "../components/StateStates";
import { analyze, getModels, loadModels } from "../api/client";
import type { AnalyzeRequest, ModelsStatusOut } from "../api/types";
import { stashResult } from "../App";

export default function LandingPage() {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelsStatusOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);

  useEffect(() => {
    getModels().then(setModels).catch(() => setModels(null));
  }, []);

  const runAnalysis = useCallback(
    async (req: AnalyzeRequest) => {
      setLoading(true);
      setError(null);
      try {
        const result = await analyze(req);
        stashResult(result);
        navigate("/results", { state: { result } });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Analysis failed.");
      } finally {
        setLoading(false);
      }
    },
    [navigate]
  );

  const handleConfirmLoad = useCallback(async () => {
    setModalLoading(true);
    setModalError(null);
    try {
      const status = await loadModels(true);
      setModels(status);
      setModalOpen(false);
    } catch (e) {
      setModalError(e instanceof Error ? e.message : "Failed to load models.");
    } finally {
      setModalLoading(false);
    }
  }, []);

  const semanticReady = models?.embedding.loaded && models?.nli.loaded;

  return (
    <div className="mx-auto flex min-h-full max-w-3xl flex-col px-6 py-16">
      <header className="text-center">
        <h1 className="text-4xl font-bold tracking-tight text-ink">
          Paper<span className="text-accent">DNA</span>
        </h1>
        <p className="mt-3 text-lg text-slate-600">
          Trace any scientific claim to its supporting evidence.
        </p>
      </header>

      <main className="mt-10">
        <SearchForm onSubmit={(req) => void runAnalysis(req)} loading={loading} />

        {error && (
          <div className="mt-4">
            <ErrorState message={error} />
          </div>
        )}

        {/* Model status banner */}
        <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-600">
          {models == null ? (
            <span className="text-slate-400">Checking model status…</span>
          ) : semanticReady ? (
            <span data-testid="models-ready">
              Semantic search and relationship prediction are{" "}
              <strong className="text-green-700">active</strong>.
            </span>
          ) : (
            <div className="flex items-center justify-between gap-3">
              <span data-testid="models-off">
                Running on the offline lexical fallback. ML models are{" "}
                <strong>optional</strong> and never downloaded automatically.
              </span>
              <button
                onClick={() => setModalOpen(true)}
                className="shrink-0 rounded-md border border-accent px-3 py-1.5 text-xs font-medium text-accent hover:bg-blue-50"
              >
                Enable ML models
              </button>
            </div>
          )}
        </div>
      </main>

      <ModelOptInModal
        open={modalOpen}
        embeddingMB={models?.embedding.downloadMB ?? null}
        nliMB={models?.nli.downloadMB ?? null}
        loading={modalLoading}
        error={modalError}
        onConfirm={handleConfirmLoad}
        onDecline={() => setModalOpen(false)}
      />
    </div>
  );
}
