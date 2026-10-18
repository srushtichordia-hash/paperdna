# PaperDNA demo workflow

A reproducible, fully local walkthrough. No deployment, no accounts, no
paid services. Two terminals:

## 1. Backend (FastAPI)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"
uvicorn paperdna.api.app:app --reload --port 8000
```

Verify: `curl http://127.0.0.1:8000/api/health` returns
`{"status": "ok", ...}`.

## 2. Frontend (React + TypeScript + Vite)

```bash
cd web
npm install
npm run dev
```

Open <http://localhost:5173>. The dev server proxies `/api` to the
backend on port 8000, so no CORS configuration is needed for local use.

## 3. Walkthrough

1. **Landing page** — the search field accepts a *claim* (default) or,
   via **Paste an abstract instead**, a full abstract. Try:

   > Retrieval augmentation improves factuality in closed-book question answering.

2. **Analyze Research** — the results page shows, in order:
   - the **capabilities banner** (what this run could and could not do);
   - **ranked papers** with scores and the ranking kind
     (`semantic` vs `lexical` — never conflated);
   - **predicted relationships** with label, confidence, uncertainty,
     condition context, and expandable verbatim evidence;
   - the **knowledge graph** (clickable nodes, labeled edges).

3. **Model status (honest degradation)** — with no ML extras installed,
   the app runs on the offline lexical fallback and *says so*: rankings
   are labeled `lexical`, relationships are empty with an explanatory
   message, and the graph still shows papers, claims, and conditions.
   The **Enable ML models** button opens an explicit opt-in dialog; a
   model is only ever downloaded after the user confirms.

4. **Optional: real models** (explicit download, several hundred MB):

   ```bash
   pip install -e ".[embed,nli]"
   ```

   Restart the backend, click **Enable ML models → Download & load**,
   and re-run the same query: rankings become `semantic`, and
   support/contradiction/neutral relationships appear with
   uncertainty flags.

## Screenshots

Captured from the local app (offline/degraded mode, honestly labeled):

| Screenshot | What it shows |
|---|---|
| [`landing.png`](screenshots/landing.png) | Landing page with claim/abstract modes |
| [`model-opt-in.png`](screenshots/model-opt-in.png) | Explicit model download confirmation dialog |
| [`results-papers.png`](screenshots/results-papers.png) | Capabilities banner + ranked papers (lexical badges) |
| [`results-relations.png`](screenshots/results-relations.png) | Relations section (empty-state honesty in degraded mode) |
| [`knowledge-graph.png`](screenshots/knowledge-graph.png) | Knowledge graph with node detail panel |
| [`validation-error.png`](screenshots/validation-error.png) | Inline validation feedback on empty claim |

## Related documentation

- [Evaluation methodology and results](evaluation.md)
- [Release checklist](release-checklist.md)
