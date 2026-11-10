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

Responses include the query, a conservative answer or abstention, uncertainty, provider statuses, a medical-use disclaimer, and ranked evidence records with source IDs, publication metadata, and URLs. The MVP deliberately does not synthesize clinical recommendations; callers should review cited source material directly. Provider failure is reflected in `provider_status` and does not prevent the remaining provider from returning results.

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

## Tests

Tests use local fake providers and do not make external API calls:

```powershell
pytest
```

They cover health routes, validation, failure behavior, missing optional credentials, normalization, deduplication, ranking, citations, and abstention.

## Current limitations

- PubMed search uses ESearch, ESummary, and EFetch; abstracts may be absent in PubMed records.
- Perplexity search is broad web search and is not itself a medical authority. Source quality is exposed for caller review; it does not establish that a source supports any particular claim.
- No automated claim-to-passage entailment verification, clinical conflict analysis, or validated grading methodology is implemented.
- The current response reports retrieved records and does not produce clinical treatment advice.
- Translation is limited to short response framing; it does not translate source records.
- This service is not for emergencies or individual treatment decisions.

## Roadmap

- **Phase 2:** Europe PMC, official guideline sources, stronger citation verification, question-specific ranking.
- **Phase 3:** PostgreSQL, Qdrant, document ingestion, source versioning, provenance tracking.
- **Phase 4:** Medical knowledge graph, validated evidence grading, conflict detection, freshness monitoring.
- **Phase 5:** Persian clinical interface, web application, Telegram and mobile clients, subscription API.
