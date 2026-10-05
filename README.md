# AI Research Radar

**Find the AI papers worth reading, explain why they matter, and leave the editorial judgment to Michał.**

The primary product is **one Radar per ISO week**, with **Best Week → Best Month →
Best Year** drilldowns. This Python 3.12+ pipeline discovers research on arXiv, filters
for AI relevance, enriches metadata, produces deterministic rankings, and analyzes only
a small selection.
A separate critic challenges every analysis before canonical JSON and Markdown drafts
are generated.

It produces content for the existing site's **`/research`** and **`/research/<slug>`** routes.
It does not rebuild the website, touch `/blog`, impersonate Michał, or automatically
publish a personal opinion. Primary sources remain directly linked.

## Architecture

```text
arXiv Atom API → normalize/deduplicate → deterministic AI filter
  → inexpensive structured relevance/quality estimate
  → optional OpenAlex / Semantic Scholar / GitHub / Hugging Face
  → timestamped SQLite metric snapshots → deterministic scoring
  → publication-week selection (0–10 papers above a quality threshold)
  → weekly Radar → quality-weighted monthly/yearly Radars
  → select at most TOP_PAPERS total for deep analysis
  → cached PDF extraction → structured deep analysis → separate critic
  → validated JSON + Markdown drafts → draft PR → human review → existing site
```

SQLite holds operational state, not the website's content source. Generated JSON is the
canonical structured content representation. There is no database server, queue,
vector database, or separate service.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync --locked
cp .env.example .env
# Edit .env locally; never commit it.
uv run research-radar run --top 10 --dry-run
```

The default command processes the **previous complete ISO week**, Monday through Sunday
in UTC, not a rolling seven-day window:

```bash
uv run research-radar
uv run python -m research_radar
```

On Monday 2026-10-05 this targets `2026-W40` (2026-09-28 through 2026-10-04).
`run --period 2024-W08` explicitly targets one publication week; use backfill for
historical discovery without automatic LLM spending. Current-week work is explicitly
`preview-week` and never replaces a canonical complete weekly Radar.

All integrations are optional. Without an OpenAI key, relevance is **explicitly marked
as deterministic**, unknown research-quality dimensions remain unknown, and deep
analysis is skipped rather than invented. Public metadata endpoints may work without
tokens but have lower quotas. Source outages are logged and absent measurements remain
`None`, never silently become zero.

`--dry-run` still performs reads, can spend money on configured LLM calls, and writes
local SQLite/cache/JSON/Markdown. It suppresses GitHub writes; it is **not** an offline
mode or a no-cost simulation. Tests use fixtures and do not require live APIs.

## Configuration

Settings are validated by `pydantic-settings`; environment variables override `.env`.

| Variable | Default / purpose |
|---|---|
| `OPENAI_API_KEY` | Optional; enables structured relevance, analysis, and critic calls |
| `GITHUB_TOKEN` | Optional; authenticated repository metadata and explicitly enabled PR writes |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional; improves metadata API access |
| `HUGGINGFACE_TOKEN` | Optional; authenticated Hugging Face access |
| `OPENALEX_EMAIL` | Optional; OpenAlex contact parameter |
| `RESEARCH_DAYS` | `7`, legacy standalone/rolling discovery window; does not change weekly `run` |
| `TOP_PAPERS` | `10`, total deep-analysis budget, not ten per ranking |
| `WEEKLY_MIN_SCORE` | `70`, minimum supported publication/retrospective score for selection |
| `WEEKLY_MAX_PAPERS` | `10`, maximum selected papers per week; not a quota |
| `WEEKLY_MIN_PAPERS` | `0`, optional soft warning only; never inserts below-threshold papers |
| `PUBLICATION_PROXY` | `true` for normal recent-week runs; historical backfill defaults to strict unless its explicit flag is used |
| `RADAR_SCORING__WEEKLY_COEFFICIENTS` | `[1,0.7,0.5,0.35,0.2]`, diminishing slots plus a mean tail |
| `RADAR_SCORING__MONTHLY_COEFFICIENTS` | `[1,0.7,0.5,0.2]` |
| `RADAR_SCORING__YEARLY_COEFFICIENTS` | `[1,0.7,0.5,0.35,0.2]` |
| `RADAR_SCORING__PEAK_SHARE` | `0.8`, best-child contribution |
| `RADAR_SCORING__QUALITY_FLOOR`, `RADAR_SCORING__QUALITY_POWER` | `70`, `2`, quality-tail transformation |
| `RETROSPECTIVE_WEIGHTS` | Research quality `60%`, citations `25%`, GitHub adoption `10%`, community `5%`; nested component overrides supported |
| `ARXIV_CATEGORIES` | `cs.AI,cs.LG,cs.CL,cs.CV,stat.ML` |
| `RELEVANCE_THRESHOLD` | `60` |
| `DRY_RUN` | `true`; disables remote mutations |
| `RELEVANCE_MODEL` | `gpt-4.1-mini` |
| `ANALYSIS_MODEL`, `CRITIC_MODEL` | `gpt-4.1` |
| `CONCURRENCY` | `4`, bounded workers |
| `DATA_DIR` | `data` |
| `DATABASE_PATH` | `data/radar.sqlite3` |
| `CACHE_DIR` | `.cache/research-radar` |
| `HTTP_CACHE_TTL` | `21600` seconds |
| `ARXIV_REQUEST_INTERVAL` | `3.1` seconds, minimum 3 seconds |
| `MAX_PDF_BYTES` | `30000000`, bounded PDF download |
| `MAX_ANALYSIS_CHARACTERS` | `100000`, primary-text input budget |
| `GITHUB_REPOSITORY` | Optional `owner/repository`, required for local PR submission |
| `GITHUB_BASE_BRANCH` | `master`; set to the target repository's actual default branch |

Empty optional credential values are treated as unavailable. Secrets are never
hard-coded, included in generated content, or deliberately written to logs.

## Weekly commands and historical exploration

```bash
# Primary product: previous complete week; current week is an incomplete preview.
uv run research-radar run --dry-run
uv run research-radar preview-week --top 10

# Historical discovery: resumable, strict cutoff scoring, no new LLM calls by default.
uv run research-radar backfill --from 2023-01-01 --to 2024-12-31 --workers 4
uv run research-radar backfill --year 2024 --skip-enrichment --resume
uv run research-radar backfill --year 2024 --no-resume

# Read-only best-period tables; ISO year filtering for weekly Radars.
uv run research-radar best-week --year 2024 --limit 20
uv run research-radar best-month --from 2023-01-01 --to 2024-12-31
uv run research-radar best-year --mode retrospective
uv run research-radar calibration --year 2024

# Recalculate from existing local state only: no discovery, HTTP, or LLM calls.
uv run research-radar retrospective
uv run research-radar retrospective --year 2024
uv run research-radar best-week --year 2024 --mode retrospective

# Explicit selected analysis: one GLOBAL budget, not ten for each of twenty weeks.
uv run research-radar analyze-selected --period 2024-W08 --top 10
uv run research-radar analyze-selected --best-weeks 20 --top 10
uv run research-radar backfill --year 2024 --analyze --confirm-expensive
uv run research-radar backfill --year 2024 --classify --confirm-expensive
```

`--period` also accepts `2024W08`, normalized to `2024-W08`. Date ranges are inclusive,
and `--to` defaults to today UTC for backfill/retrospective. Backfill requires `--from`
or `--year`; `--year` is a calendar-year discovery range and cannot be combined with
`--from`/`--to`. Retrospective defaults to 2023-01-01 through today, or accepts the same
range/year options. Partial boundary weeks retain explicit coverage and completeness
flags. Best-week's `--year` uses the **ISO weekly year**, including weeks crossing
calendar-year boundaries; best-month/year use their hierarchy IDs. Month/year date
filters include overlapping calendar periods.

Best-period tables show stable-ID tie-breaking, effective mode-specific scores, date
ranges, selected counts, the top paper and its score, explanation, completeness, and
drilldown IDs. Months reference their best week; years reference their best month and
best weeks. Unknown scores are listed separately, never ranked as zero. Calibration
reports weeks with **0, 1, 2–3, 4–9, or ≥10** selected papers, with unknown-score weeks
excluded from those bins and counted separately. A zero-paper week is valid; no weekly
minimum is a forced publication quota.

Backfill resumes per-week discovery, relevance, enrichment, ranking, and analysis
checkpoints. Failed/incomplete phases remain resumable; `--no-resume` requests
reprocessing. A namespace binds the start date, categories, scoring/classification and
enrichment/proxy options; extending the end date can reuse completed discovery while
relevant configuration changes get a distinct namespace. Enabling analysis does not
discard discovery checkpoints. It reuses
cached/historical relevance or deterministic filtering by default and makes **zero new
LLM calls even when `OPENAI_API_KEY` is present**. Metadata API usage can still be large;
use `--skip-enrichment` to conserve provider quotas. `--skip-analysis` is the default.
`--analyze` (alias `--analyze-selected`) enables selected deep analysis, and `--classify`
enables inexpensive LLM classification; either requires `--confirm-expensive` before
processing begins. Critic retries may multiply the calls within the global selection
budget. `analyze-selected` itself is an explicit opt-in; budgets over ten also require
confirmation. Best tables and calibration only query period Radars, not the full paper
corpus.

## Independent legacy stages

```bash
uv run research-radar discover --days 7
uv run research-radar enrich
uv run research-radar rank --top 10
uv run research-radar analyze --top 10
uv run research-radar generate
uv run research-radar validate
```

Each stage consumes persistent state left by preceding stages. Discovery merges new
versions rather than creating duplicate papers. Enrichment includes relevant historical
records so an older paper can resurface when attention increases. `analyze` reads the
last saved selection. `generate` refuses stale analysis whose evidence no longer
matches the paper record.

These low-level per-paper capabilities remain available. The explicit
`run --days 7 --top 10 --dry-run` compatibility command logs a deprecation notice and
uses the legacy rolling pipeline; it is not the primary weekly product. Omit `--days`
to process the previous complete ISO week. `discover --days 7` remains independently
supported.

Each run prints discovery/relevance/enrichment/ranking/analysis/critic counts and
records execution metadata in SQLite. Weekly runs display the scoped week and its
month/year hierarchy rather than unrelated global latest paper rankings; legacy runs
retain their Trending and Important lists. Expected per-paper failures do not
abort other papers; reports distinguish successful, partial, and failed runs.
Programming errors and cancellation are not disguised as successful processing.

To explicitly submit a **draft review PR** from a local run:

```bash
# Set GITHUB_REPOSITORY, GITHUB_BASE_BRANCH and GITHUB_TOKEN first.
uv run research-radar run --no-dry-run
```

Only generated research/Radar JSON, draft Markdown, and rankings can be submitted. Local
submission uses GitHub's Git Data/PR REST APIs, not a commit of the entire working tree.
It never approves, merges, or publishes content.

## Data and editorial workflow

```text
data/papers/YYYY/MM/<slug>.json                   canonical ResearchPaper
data/articles/drafts/YYYY-MM-DD-<slug>.md         technical reading draft
data/rankings/YYYY-MM-DD.json                    ranking explanations/selection
data/radars/weeks/YYYY/YYYY-Www.json              canonical publication-week Radar
data/radars/months/YYYY/YYYY-MM.json              best-week references and month score
data/radars/years/YYYY.json                       best-month/week references and year score
data/radars/retrospective/                       separate present-day reassessment hierarchy
data/radars/previews/                            explicitly incomplete current-week previews
data/articles/published/                        reserved for human publication
data/radar.sqlite3                              ignored operational state
.cache/research-radar/                           ignored API/PDF/LLM caches
```

Every Radar JSON has adjacent `.md` reading/review content. Preview weeks use
`data/radars/previews/weeks/YYYY/YYYY-Www.{json,md}`; retrospective documents mirror
the weeks/months/years hierarchy beneath `data/radars/retrospective/`.

Slugs include a short stable paper-ID hash to avoid title collisions, and existing
slugs are retained when a paper's title changes. Pydantic models use explicit fields,
reject unexpected generated properties, and distinguish missing values from observed
zero counts.

`ResearchPaper` contains bibliographic metadata, primary links, optional metrics,
score dimensions and breakdowns, all required research sections, a supported timeline,
section-level source references, reproducibility classification, full-text availability,
prompt versions, model names, critic issues, and human editorial state.

The fixed reading template covers TL;DR, practical motivation, research timeline,
problem, mechanism, novelty, baselines, exact available results, comparison fairness,
authors' versus analysis-derived limitations, reproducibility, implications, and
clearly labeled future speculation. Unknown information should say **Unknown**.

Every generated draft starts in `review` if the critic passed, otherwise `needs_review`.
The critic can request up to **two regenerations** (three attempts total). Unresolved
high-severity issues cannot enter approved/published states.

**Michał's take is never part of an LLM output schema.** Its JSON field is initially
`null`; its Markdown block is empty:

```markdown
## Michał's take

<!-- HUMAN EDITORIAL SECTION -->

<!-- END HUMAN EDITORIAL SECTION -->
```

Write the editorial perspective between those markers. Regeneration preserves that
block and transfers its contents into `michaels_take`. Human edits to the JSON field
are also preserved. Questions for Michał may be generated separately, but never
answered for him. Already approved/published JSON is not overwritten by automation.
If JSON and Markdown contain different nonempty editorial takes, generation preserves
both and reports a conflict; align them manually before regenerating.
Human editing outside the editorial block is not automatically merged; mark content
approved when it should be protected.

The frontend can load `data/papers/**/*.json`, list papers at `/research`, and resolve
details using `slug`. It must explicitly decide which reviewed statuses to expose.
Treat generated Markdown as untrusted content: disable raw HTML or sanitize it in the frontend.
`needs_review` content is not ready for publication. Merging a content PR by itself
does not implement or invoke website publication in this repository.

## Radar hierarchy and time-aware scoring

ISO weeks belong to the month containing their **Thursday**. This deterministic
ownership rule assigns every canonical week to exactly one month; cross-year weeks
retain their ISO year while monthly ownership follows that Thursday. Previews are
separate, incomplete documents, never eligible for canonical best-week/month/year
rankings or approved/published status.

Weekly selections include only supported scores at or above `WEEKLY_MIN_SCORE`, up to
`WEEKLY_MAX_PAPERS`. Unknown scores do not become weak papers. Monthly scores aggregate
strong **weekly Radar scores**, not paper quantities; yearly scores aggregate strong
**monthly Radar scores**, not publication counts.

For descending known child scores `s`, transform each with:

```text
q(s) = max(0, (s - quality_floor) / (100 - quality_floor)) ** quality_power
weighted_q = Σ coefficients[i] * q(s[i])       # first len(coefficients)-1 slots
           + coefficients[-1] * MEAN(q(remaining tail))
aggregate = peak_share * best_child_score
          + (1 - peak_share) * 100 * weighted_q / SUM(coefficients)
```

The denominator is fixed; unused slots contribute nothing and an empty tail has no
contribution. The final coefficient multiplies the **mean tail**, never its count.
Defaults are weekly/yearly `[1,.7,.5,.35,.2]`, monthly `[1,.7,.5,.2]`, `peak_share=.8`,
`quality_floor=70`, and `quality_power=2`. Thus two excellent papers can beat ten
mediocre ones. Unknown child scores are omitted, not converted to zero, and their
counts and completeness remain explicit. No known historical evidence means an
unknown aggregate, not an invented weak week.

**Publication mode** answers what could have been scored by the end of the original
publication week. Publication time and observation time are distinct: metrics,
relevance/quality judgments, primary-document versions, and velocity baselines must
be eligible at the explicit data cutoff. Future citations, adoption, or later paper
revisions must not leak into the historical publication score. Missing cutoff-eligible
observations remain `None`, with known/unknown coverage exposed.

Historical backfill is strict by default. Its optional `--publication-proxy` may use
currently estimated research-quality dimensions only from an unchanged primary
document already available at the cutoff. These are labelled
`publication_document_proxy`: **approximate estimates, not authentic historical
observations**, and a modern LLM's pretraining may itself contain future knowledge.
Revised papers without the old version remain unknown; the latest revision is not
evidence for a past cutoff. Normal recent-week runs enable this labelled proxy by
default because enrichment usually happens after the completed week's cutoff.

**Retrospective mode** answers which papers from a publication week proved valuable
using observations available at the present reassessment cutoff. It uses existing
local state without automatic network/LLM calls. Its `retrospective_score` is distinct
from `publication_score`/`radar_score`, stored under separate output paths and database
mode identities. It must never overwrite or be presented as the original publication
ranking.

### Existing deterministic per-paper scores

The LLM estimates only relevance and research-quality dimensions. Every quality
estimate has a bounded score, a reason, and confidence. Abstract-only estimates are
provisional, not a substitute for experimental review. The LLM never chooses final
ranking values.

| Importance component | Weight |
|---|---|
| Scientific impact | 30% |
| Novelty | 25% |
| Empirical quality | 20% |
| Practical relevance | 10% |
| Citation signal | 5% |
| Venue | 5% |
| Author/lab signal | 5% |

| Trending component | Weight |
|---|---|
| GitHub velocity | 30% |
| Community attention | 20% |
| Hugging Face velocity | 15% |
| Citation velocity | 15% |
| Freshness | 10% |
| Absolute attention | 10% |

| Overall component | Weight |
|---|---|
| Scientific impact | 25% |
| Novelty | 20% |
| Empirical quality | 15% |
| Practical relevance | 10% |
| GitHub velocity | 10% |
| Community interest | 10% |
| Citation velocity | 5% |
| Venue/author signal | 5% |

Missing components stay unknown. Scores expose **coverage** and their weighted
contributions; a high score based on sparse evidence should not be mistaken for a
well-supported result. Raw attention counts and observed velocities are normalized
by fixed, deterministic functions rather than by LLM judgment.

Observed components are reweighted by their available weight. Importance and Overall
require at least 50% coverage and supported scientific-quality evidence; Trending
requires at least 30% coverage and observed attention. Below those thresholds, the
score is `None`, not a number based on freshness alone.

Raw counts and daily rates use `100 × x / (x + h)` (nonpositive growth scores zero).
Half-saturation points `h` are 250 GitHub stars/day, 5 citations/day, 25 HF votes/day,
5,000 absolute GitHub stars, 100 citations, and 100 HF votes. GitHub's 24h and 7d
daily-rate signals blend 70:30 when both are available. Community attention averages
available HF vote and observed trending signals; absolute attention averages available
normalized star/citation/vote counts. Overall community interest averages available
community, HF growth, and absolute attention signals.

Freshness uses elapsed whole days, with multipliers `1.50` for 0–2, `1.25` for 3–7,
`1.10` for 8–14, `1.00` for 15–30, and `0.80` thereafter. The freshness component is
`100 × multiplier / 1.50`; it occupies its configured Trending weight rather than
multiplying an entire final score. Age is not a hard exclusion.
Absolute stars have only a small weight; a low-star repository growing quickly can
outrank a large repository growing slowly.

SQLite snapshots retain timestamped citations, stars, upvotes, and series provenance.
Velocity requires sufficient history and compatible source identities; the first
observation does not invent a growth rate. Citation counts from indexes are not summed.
Rates are signed counts/day, including the 7d GitHub field. Baselines must be
24–36 hours old for daily signals and 7–8 days old for the weekly signal.

Weights are Pydantic settings. Each group must be nonnegative and sum to one. For example:

```env
WEIGHTS__IMPORTANCE__SCIENTIFIC_IMPACT=0.35
WEIGHTS__IMPORTANCE__NOVELTY=0.20
```

Those two changes preserve the default group's total. All other components are also
configurable through the same nested environment-variable convention.

For deterministic backtesting, call the ranking engine with historical `PaperRecord`
values, historical snapshots, and an explicit `now`. There is no learned ranking
model or dependence on the current wall clock in the scoring API. Future backtests
must avoid using metadata or LLM judgments that were unavailable at the simulated date.

## Evidence, sources, and costs

The authoritative input is the original paper, not a community post or README.
Analysis receives the title, abstract, extracted paper text when available, metadata,
repository evidence/README, metrics, and related works. Related metadata does not
establish historical lineage by itself.

GitHub associations require explicit paper-linked evidence or a README that references
the paper's identifier; title resemblance alone is not sufficient. Unverified
repositories stay unassociated. Metrics depend on what the public provider actually
returns; unavailable counts, trends, models, or datasets remain unknown.

arXiv uses its official Atom API, pagination, and a rate-limited request schedule.
Transient network/429/server failures are retried. Enrichment requests are bounded.
PDF downloads are size-limited and text extraction uses PyMuPDF. Failed extraction
falls back to the abstract and visibly reduces evidence confidence.

Caches cover API responses, PDF bytes/text, and structured LLM responses. LLM cache
keys incorporate paper/evidence content, prompt text/version, schema, and model.
Changing a prompt invalidates its generated response cache. Identical evidence is
not intentionally reanalyzed on every run.
Authenticated metadata responses are not disk-cached, preventing token-scoped data
from being reused under different credentials. Relevance is rechecked through its
cache each run, so changing its prompt or model does not leave stale persisted scores.

Only metadata/abstracts reach the inexpensive relevance model. Only selected papers
reach deep analysis. Critic/regeneration calls can increase cost; `TOP_PAPERS`,
model selection, caching, and the relevance threshold are the main controls.

## Prompts and providers

Versioned templates live in `prompts/` for checkout customization and under
`src/research_radar/prompts/` for packaged operation. Keep both copies aligned when
changing defaults. Set `RELEVANCE_PROMPT_VERSION`, `ANALYSIS_PROMPT_VERSION`, and
`CRITIC_PROMPT_VERSION` when intentionally changing prompt behavior.

The analysis and critic must avoid invented authors, numbers, citations, repositories,
historical relationships, and unsupported SOTA claims. They distinguish **Fact**,
**Interpretation**, and **Speculation** and treat embedded source text as untrusted
data, not instructions. Deterministic source validation supplements the LLM critic.
An LLM critic is still not a guarantee of factual correctness; human review is required.

To add an LLM provider, implement `LLMProvider.generate_structured` in
`analysis/llm.py`'s provider-agnostic contract, returning a schema-validated Pydantic
model, then inject it into `RelevanceFilter` and `AnalysisService`. Keep the critic
as a separate invocation and preserve cache/provenance semantics.

To add a metadata source, create an async collector, validate its API payload,
map known measurements into explicit models, and wire it into `EnrichmentPipeline`.
Add source references and fixture-driven failure/missing-data tests. Do not introduce
title-only identity guesses or invent zeros for unavailable measurements.

## GitHub Actions

`.github/workflows/research-radar.yml` runs weekly on **Monday at 07:00 UTC** and on manual
dispatch. It installs uv/Python, restores operational history and caches, runs quality
checks, executes `run` without `--days` for the previous complete ISO week, validates
generated JSON, uploads Radar and selected-paper content artifacts,
and creates or updates a **draft PR** on `research-radar/automated-drafts`.
The weekly job allows 240 minutes for discovery/abstract processing but keeps the
global deep-analysis selection budget at ten.

`.github/workflows/research-radar-backfill.yml` is **manual dispatch only**, never part
of the weekly cron. Inputs are `from_date` (default `2023-01-01`), optional `to_date`
(empty means today UTC), `workers` (bounded choice, default four), `skip_analysis`
(default true), `skip_enrichment` (default false), `classify` (default false),
`confirm_expensive` (default false), `publication_proxy` (default false), and `create_pr`
(default false). Unconfirmed analysis/classification is rejected before pipeline
initialization. The job resumes by default, permits up to 360 minutes, and stores
JSON/Markdown plus a SQLite operational checkpoint backup as artifacts even on partial
failure. PR creation is optional and always draft-only; no workflow approves, merges,
or publishes a Radar.

Configure Actions secrets:

```text
OPENAI_API_KEY
SEMANTIC_SCHOLAR_API_KEY
HUGGINGFACE_TOKEN
OPENALEX_EMAIL
```

GitHub supplies `GITHUB_TOKEN` automatically. Enable **Allow GitHub Actions to create
and approve pull requests** in repository Actions settings; the workflow requests
contents/PR write permissions but does not approve anything. Organization policies
may require a separately supplied token or forbid automated PR creation.

The workflow prepares the existing review branch before generation, so its editorial
block edits are retained. It merges the latest default branch without force-pushing;
merge conflicts fail visibly instead of overwriting human changes. Manual dispatch
can disable PR creation while still retaining artifacts.

Both workflows use the `research-radar` concurrency group to serialize shared SQLite
history, with unique weekly/backfill cache keys and a shared `radar-state-<OS>-`
restore prefix. Explicit cache save steps run even after pipeline failure, preserving
checkpoints (including an unfinished SQLite WAL) for the next resume. Pipeline step
timeouts leave room inside the job timeout for state/artifact cleanup. Artifacts include
a consistent SQLite backup but
never `.env`, tokens, or the request/LLM cache. Only scoped generated
`data/radars`, `data/papers`, `data/rankings`, and `data/articles/drafts` are committed.
The review branch is prepared before generation even when only artifacts are requested,
so human editorial blocks on that branch are retained.

SQLite/cache history uses Actions cache restore keys across runs. Cache eviction or
deletion resets available velocity history; missing history stays unknown. This is
an MVP operational convenience, not a durable database backup strategy.

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
uv run research-radar validate
uv build
```

Tests use fixtures, mock HTTP, and mock structured providers; they require no keys,
paid calls, or live discovery. They cover identity normalization/version deduplication,
freshness, historical velocity, weighting/rank stability, absent metadata, malformed
structured responses, critic retries, editorial preservation, SQLite, and full pipeline
integration.

## Troubleshooting

| Symptom | What to inspect |
|---|---|
| No deep-analysis drafts | Check `OPENAI_API_KEY`, relevance threshold, selected rankings, and logged paper failures |
| Velocity is unknown | Collect snapshots across the required horizon; confirm source/repository identity is unchanged |
| Missing metadata | Provider quota/authentication/outage, identifier availability, or lack of official repository evidence |
| Abstract-only analysis | PDF URL, download limit, extraction errors, or text-input limit; confidence should be reduced |
| `needs_review` | Read stored critic issues, check primary evidence, and resolve them through human review |
| Invalid weights/settings | Each weight group must sum to one; categories must be comma-separated arXiv codes |
| GitHub PR rejected | Token scope, repository name/base branch, Actions PR policy, and branch conflicts |
| Stale analysis rejected | Run `analyze` after changing the evidence, then `generate` |
| Unexpected repeat API usage | Cache location/permissions, TTL, prompt versions, model changes, or changing evidence |

Accuracy takes precedence over fluent prose. This is a research discovery and analysis
product, not an automated content farm or a replacement for reading the paper.