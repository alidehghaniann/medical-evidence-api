# Medical Evidence API

An independent, modular service for retrieving medical literature and presenting source-traceable records. The core rule is **no evidence, no clinical answer**: the MVP abstains when it cannot return attributable evidence. It does not diagnose or prescribe and is not a validated clinical decision support system.

## Architecture

```text
FastAPI routes -> search service -> provider interface
                                  |-> PubMed E-utilities
                                  `-> optional Perplexity Agent API
       evidence normalization -> deduplication -> heuristic ranking -> citation checks
```

Providers implement a shared asynchronous interface and can fail independently. PubMed uses NCBI E-utilities; Perplexity is optional and only enabled when `PERPLEXITY_API_KEY` is configured. Its Agent API web search results are retained with their returned URLs. Generated prose is never treated as a source record. The evidence rank is a transparent heuristic, not a validated evidence-grading system.

## Requirements and Windows installation

Python 3.10.6 is supported. In PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env` to add credentials as needed. PubMed works without an API key; an NCBI email is recommended. Perplexity is not required. Never commit `.env` or credentials.

## Configuration

| Variable | Purpose |
| --- | --- |
| `PERPLEXITY_API_KEY` | Optional Perplexity Agent API credential |
| `NCBI_API_KEY` | Optional NCBI E-utilities key for higher rate limits |
| `NCBI_EMAIL` | Contact email sent to NCBI |
| `APP_ENV` | Environment label, defaults to `development` |
| `REQUEST_TIMEOUT_SECONDS` | HTTP timeout, defaults to 15 seconds |
| `MAX_SEARCH_RESULTS` | Configured result ceiling |
| `MINIMUM_EVIDENCE_SCORE` | Abstention threshold, defaults to 0.25 |

## Run

```powershell
uvicorn app.main:app --reload
```

Interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`.

## Endpoints

### `GET /`

Returns service name, status, and docs location.

### `GET /health`

Returns `{"status":"ok"}` when the API process is responsive. It does not check upstream provider health.

### `POST /v1/medical/search`

English request:

```json
{"query":"What is the evidence about H. pylori treatment?","language":"en","limit":10}
```

Persian request:

```json
{"query":"شواهد درباره درمان هلیکوباکتر پیلوری چیست؟","language":"fa","limit":10}
```

Responses include the query, a conservative status, provider statuses, a medical-use disclaimer, and ranked evidence records with source IDs, publication metadata, and URLs. `/search` returns retrieved records without a synthesized recommendation. `/ask` can synthesize a cited informational response from those records; callers should review the source material directly. Provider failure is reflected in `provider_status` and does not prevent the remaining provider from returning results.

Example response (truncated):

```json
{
  "query": "What is the evidence about H. pylori treatment?",
  "language": "en",
  "answer": "Retrieved sources are listed below. This MVP does not generate a clinical recommendation; review the cited evidence directly.",
  "abstained": false,
  "uncertainty": "Evidence ranking is heuristic and has not been clinically validated.",
  "evidence": [
    {
      "source": "pubmed",
      "source_id": "12345678",
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

This endpoint reuses the evidence retrieval service, then asks the configured answer provider to synthesize only from those retrieved records. Perplexity Agent API is currently the optional answer provider; it is called without search tools. If no answer provider is configured, the endpoint returns HTTP 503.

Request:

```json
{"query":"What is the first-line treatment for GERD?","language":"en"}
```

Persian requests use `"language":"fa"` and Persian query text.

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

## Local MCP server

The local stdio MCP server exposes `medical_search` (`query`, optional `language`) and `medical_ask` (`query`, optional `language`). Both return typed, machine-readable response models and call the same Medical Evidence Core as REST; MCP does not bypass citation validation, claim verification, or abstention. Run it from the repository root with:

```powershell
python -m app.mcp_server
```

## Medical Agent

The lightweight CLI routes evidence-search requests to `medical_search` and medical synthesis questions to `medical_ask`. It returns the selected tool's typed result unchanged, preserves abstention, and does not generate medical answers on its own. Clearly non-medical questions are declined without starting the MCP server. Example:

```powershell
python -m app.medical_agent "Find recent guidelines about asthma"
```

## Current limitations

- PubMed search uses ESearch, ESummary, and EFetch; abstracts may be absent in PubMed records.
- Perplexity search is broad web search and is not itself a medical authority. Source quality is exposed for caller review; it does not establish that a source supports any particular claim.
- No automated claim-to-passage entailment verification, clinical conflict analysis, or validated grading methodology is implemented.
- Conflicting evidence is not systematically detected; the answer flow does not establish consensus across retrieved sources.
- `/search` returns evidence only; `/ask` produces informational synthesis but does not provide individualized treatment advice.
- Claim support checks use an LLM and are not a validated medical fact-checking method; errors and low-confidence checks fail closed as uncertain.
- Translation is limited to short response framing; it does not translate source records.
- This service is not for emergencies or individual treatment decisions.

## Roadmap

- **Phase 2:** Europe PMC, official guideline sources, stronger citation verification, question-specific ranking.
- **Phase 3:** PostgreSQL, Qdrant, document ingestion, source versioning, provenance tracking.
- **Phase 4:** Medical knowledge graph, validated evidence grading, conflict detection, freshness monitoring.
- **Phase 5:** Persian clinical interface, web application, Telegram and mobile clients, subscription API.
