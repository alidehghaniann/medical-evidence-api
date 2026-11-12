# Medical Evidence API

An independent, modular service for retrieving medical literature and presenting source-traceable records. The core rule is **no evidence, no clinical answer**: the MVP abstains when it cannot return attributable evidence. It does not diagnose or prescribe and is not a validated clinical decision support system.

## Architecture

```text
Question
  -> Query Understanding (validated MedicalQuery; fail closed for Persian)
  -> Source Router (deterministic; configured providers)
  -> Shared Retrieval (PubMed + Europe PMC; optional Perplexity search)
  -> Normalization and deduplication (EvidenceRecord)
  -> Evidence Assessment (transparent baseline, explicit unknowns)
  -> Ranking (relevance-led deterministic score)
  -> SearchResponse
  -> Clinical Synthesis (configured AnswerProvider; ask only)
  -> Claim and Citation Verification
  -> Answer or Abstain (application policy)
```

REST, MCP, and the CLI agent use the same `MedicalSearchService` and medical application functions. Search stops at ranked evidence. Ask continues through the existing answer provider, citation mapping, claim coverage checks, and claim verifier; any unsupported or uncertain claim fails closed to abstention. Evidence assessment, source authority, recency, and ranking are engineering baselines, not clinically validated assessment methods. The assessment is not GRADE, statistical appraisal, or risk-of-bias assessment.

Providers implement shared asynchronous interfaces and can fail independently. PubMed uses NCBI E-utilities, and Europe PMC uses its REST search API. The shared evidence search uses these controlled literature sources; Perplexity web search is not added to the evidence corpus. Perplexity provides query understanding when `QUERY_UNDERSTANDING_PROVIDER=perplexity`, and provides answer synthesis and claim verification when `ANSWER_PROVIDER=perplexity`. The development-only provider returns an exact sentence from matching retrieved abstract evidence and uses an exact-text verifier; it does not synthesize, translate, browse, or add medical knowledge. The evidence rank is a transparent heuristic, not a validated evidence-grading system.

## Requirements and Windows installation

Python 3.10.6 is supported. In PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

The sample `.env` selects `ANSWER_PROVIDER=dev` and `QUERY_UNDERSTANDING_PROVIDER=ollama` for local development. The development answer provider and exact-text verifier are DEVELOPMENT/TEST ONLY and are not clinical reasoning tools. Persian query understanding uses the local Ollama HTTP API and requires `OLLAMA_MODEL` to name an installed local model; it does not require a Perplexity key. To use Perplexity, set `QUERY_UNDERSTANDING_PROVIDER=perplexity`, `ANSWER_PROVIDER=perplexity`, and provide `PERPLEXITY_API_KEY` in the ignored local `.env`. Perplexity is an intelligence layer; PubMed and Europe PMC remain the evidence sources. Production defaults to Perplexity query understanding. Production rejects the development answer provider. PubMed works without an API key; an NCBI email is recommended. Never commit `.env` or credentials.

## Configuration

| Variable | Purpose |
| --- | --- |
| `ANSWER_PROVIDER` | `dev` for deterministic extractive development/testing, or `perplexity` for real synthesis; defaults to `perplexity` |
| `PERPLEXITY_API_KEY` | Required when `ANSWER_PROVIDER=perplexity`; also enables optional Perplexity retrieval |
| `QUERY_UNDERSTANDING_PROVIDER` | `ollama` or `perplexity`; defaults to Ollama in development/test and Perplexity otherwise |
| `OLLAMA_BASE_URL` | Ollama local API base URL, defaults to `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Required when Ollama query understanding is selected; a model installed locally in Ollama |
| `NCBI_API_KEY` | Optional NCBI E-utilities key for higher rate limits |
| `NCBI_EMAIL` | Contact email sent to NCBI |
| `APP_ENV` | Environment label, defaults to `development` |
| `REQUEST_TIMEOUT_SECONDS` | HTTP timeout, defaults to 15 seconds |
| `MAX_SEARCH_RESULTS` | Configured result ceiling |
| `MINIMUM_EVIDENCE_SCORE` | Abstention threshold, defaults to 0.25 |

The default Python settings are validated and local-safe. A missing `OLLAMA_MODEL` means Persian search is reported as `query_understanding: not_configured`; it is never sent untranslated to literature providers. `/diagnostics` reports local configuration only and does not make network calls. `configured` means credentials/settings are present, not that an upstream service is reachable. Use `/health` for the lightweight process check.

## Run

```powershell
uvicorn app.main:app --reload
```

Interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`.
The minimal local evidence-search page is at `http://127.0.0.1:8000/demo`.

## Endpoints

### `GET /`

Returns service name, status, and docs location.

### `GET /health`

Returns `{"status":"ok"}` when the API process is responsive. It does not check upstream provider health.

### `GET /diagnostics`

Returns process status and non-secret provider configuration states. PubMed and Europe PMC need no credentials for basic retrieval. Query understanding and answer configuration are reported separately; a configured state does not guarantee network reachability or model availability.

### `POST /v1/medical/search`

English request:

```json
{"query":"What is the evidence about H. pylori treatment?","language":"en","limit":10}
```

Persian request:

```json
{"query":"شواهد درباره درمان هلیکوباکتر پیلوری چیست؟","language":"fa","limit":10}
```

The API detects Persian or English from the query text; the `language` field is retained for compatibility and is normalized to the detected language. Persian queries are Unicode-normalized and passed to the selected query-understanding provider for a validated, search-ready biomedical representation. In development this is local Ollama by default; in production it is Perplexity by default. If the selected provider is unavailable or returns invalid data, the request abstains and the original Persian text is not sent untranslated to literature providers. Retrieved titles and abstracts remain in their source language.

Responses include the query, a conservative status, provider statuses, a medical-use disclaimer, and ranked evidence records with source IDs, publication metadata, and URLs. `/search` returns retrieved records without a synthesized recommendation. `/ask` can synthesize a cited informational response from those records; callers should review the source material directly. Provider failure is reflected in `provider_status` and does not prevent the remaining provider from returning results.

Example response (truncated):

```json
{
  "query": "What is the evidence about H. pylori treatment?",
  "language": "en",
  "answer": "Retrieved sources are listed below. This MVP does not generate a clinical recommendation; review the cited evidence directly.",
  "abstained": false,
  "evidence_found": true,
  "evidence_count": 1,
  "abstention_reason": null,
  "uncertainty": "Evidence ranking is heuristic and has not been clinically validated.",
  "evidence": [
    {
      "source": "pubmed",
      "source_id": "12345678",
      "citation_id": "pubmed:12345678",
      "title": "Example indexed evidence title",
      "abstract": null,
      "authors": [],
      "journal": "Example Journal",
      "publication_year": 2024,
      "doi": null,
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
      "evidence_type": null,
      "retrieved_at": "2026-01-01T00:00:00Z",
      "relevance_score": 0.5,
      "authority_score": 0.55,
      "recency_score": 0.9,
      "rank_score": 0.53
    }
  ],
  "provider_status": {"pubmed": "ok"},
  "disclaimer": "This service provides research information, not diagnosis or treatment. It is not a validated clinical decision support system. Consult a qualified clinician."
}
```

### `POST /v1/medical/ask`

This endpoint reuses the evidence retrieval service and configured answer and claim-verification providers. `ANSWER_PROVIDER=perplexity` uses the Perplexity Agent API without search tools and requires `PERPLEXITY_API_KEY`. `ANSWER_PROVIDER=dev` is restricted to development/test and returns only an exact matching abstract sentence, verified by an exact-text verifier; it does not generate clinical synthesis. With the default Perplexity selection and no key, the endpoint returns HTTP 503.

Request:

```json
{"query":"What is the first-line treatment for GERD?","language":"en"}
```

Persian questions are detected from their text and passed to the answer provider with Persian as the requested response language. The development provider remains extractive and may abstain if the retrieved evidence does not include a suitable Persian sentence; it does not translate evidence.

Response:

```json
{
  "answer": "The supplied evidence supports ... [evidence-1]",
  "language": "en",
  "citations": [
    {
      "evidence_id": "evidence-1",
      "claim": "Treatment evidence",
      "title": "Retrieved source title",
      "source": "pubmed",
      "source_id": "12345678",
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    }
  ],
  "confidence": 0.7,
  "abstained": false,
  "evidence_count": 1
}
```

Citation IDs are assigned to retrieved evidence for that request. Membership is checked deterministically, then each answer claim is checked against only its cited evidence. The answer text must have exact sentence coverage by structured claims; the successful response is rendered from verified claims with their citation IDs, never from the provider's free-form answer. Unsupported or uncertain claims, missing citations, incomplete claim coverage, or verifier failure produce an abstention. Claim statuses and confidence are returned for auditing. Search-provider failure returns HTTP 503; answer-provider failure or invalid citations return HTTP 502.

## Tests

Tests use local fake providers and do not make external API calls:

```powershell
pytest
```

They cover health routes, validation, failure behavior, missing optional credentials, normalization, deduplication, ranking, citations, and abstention.

## Retrieval evaluation benchmark

The versioned dataset is `evaluation/medical_retrieval_v1.json` (22 English/Persian cases across 11 categories). Each case includes a stable ID, query, intent, expected concepts, and a relevance judgment map where available. Four English cases contain partial, manually verified PubMed PMID judgments; the remaining 18 cases are explicitly unjudged. Persian expected search representations are reference concepts for review, not relevance labels or guaranteed translations. No result is considered relevant from title keyword overlap alone.

Run the evaluation only with explicit live opt-in:

```powershell
python -m app.evaluation --live
```

This calls the existing shared search service and its configured PubMed and Europe PMC providers. Reports are written to `evaluation/results/latest.json` and `evaluation/results/latest.md` (ignored by Git). Live requests are never run by pytest. English cases use the existing deterministic English query handling. Persian cases require the configured query-understanding provider; otherwise they are reported as failed/skipped without sending Persian text untranslated. No Perplexity answer API is used.

Metrics use unique evidence identifiers, preferring PMID, then PMCID, then DOI. Duplicate identifiers are removed before taking the top K; unidentified records remain unjudged and occupy a top-K position. Precision@K is the fraction of explicitly judged retrieved items in the top K that have positive relevance; unjudged items are excluded from its denominator and judgment coverage is reported beside it. Recall@K is the retrieved known-relevant items divided by the known-relevant judgment set. F1@K is the harmonic mean of those values. MRR uses the first positive result among judged top-K results. nDCG@K uses graded relevance from 0 to 3, with unjudged documents excluded and judged ranks compressed. If a case has no relevance judgments, its metrics are null; if it has no known relevant documents, recall and nDCG are null. These partial-pool metrics are not estimates of relevance across the full literature.

The report includes dataset version and timestamp, judged/unjudged case counts, macro-averaged overall/category/language metrics, provider status and source coverage, failures, missing known relevant identifiers, and ranking errors. Since judgments are partial and the public literature changes, compare results only alongside the same dataset version, code revision, cutoff K, and query-understanding configuration. This benchmark evaluates retrieval behavior only; it is not clinical validation.

## Local MCP server

The local stdio MCP server exposes `medical_search`, `medical_ask`, `pubmed_search`, and `europepmc_search` (`query`, optional `language`). The first two return typed results from the Medical Evidence Core; MCP does not bypass citation validation, claim verification, or abstention. `pubmed_search` uses NCBI E-utilities, and `europepmc_search` uses the Europe PMC REST API. Both source tools return normalized publication metadata and abstracts without downloading arbitrary full-text articles. Run the server from the repository root with:

```powershell
python -m app.mcp_server
```

## Medical Agent

The lightweight CLI routes explicit Europe PMC requests to `europepmc_search`; other explicit study, research, paper, PubMed, clinical trial, recent literature, and scientific evidence requests go to `pubmed_search`. Broader source discovery goes to `medical_search`, and synthesis questions go to `medical_ask`. It returns the selected tool's typed result unchanged, preserves abstention, and does not generate medical answers on its own. Clearly non-medical questions are declined without starting the MCP server. Example:

```powershell
python -m app.medical_agent "Find recent guidelines about asthma"
```

## Current limitations

- PubMed search uses ESearch, ESummary, and EFetch; abstracts may be absent in PubMed and Europe PMC records.
- Perplexity search is broad web search and is not itself a medical authority. Source quality is exposed for caller review; it does not establish that a source supports any particular claim.
- No automated claim-to-passage entailment verification, clinical conflict analysis, or validated grading methodology is implemented.
- Conflicting evidence is not systematically detected; the answer flow does not establish consensus across retrieved sources.
- `/search` returns evidence only; `/ask` produces informational synthesis but does not provide individualized treatment advice.
- Persian search uses the configured Ollama or Perplexity provider to prepare a search-ready query. Query understanding and evidence ranking are not validated medical systems.
- Claim support checks use an LLM and are not a validated medical fact-checking method; errors and low-confidence checks fail closed as uncertain.
- This service is not for emergencies or individual treatment decisions.

## Roadmap

- **Phase 2:** Official guideline sources, stronger citation verification, question-specific ranking.
- **Phase 3:** PostgreSQL, Qdrant, document ingestion, source versioning, provenance tracking.
- **Phase 4:** Medical knowledge graph, validated evidence grading, conflict detection, freshness monitoring.
- **Phase 5:** Persian clinical interface, web application, Telegram and mobile clients, subscription API.
