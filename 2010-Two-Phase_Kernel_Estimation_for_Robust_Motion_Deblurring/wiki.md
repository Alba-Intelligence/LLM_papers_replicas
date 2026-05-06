You are working in a different repository.

Your goal is to recreate, adapt, and initialize the same private Karpathy-style LLM wiki architecture that we use in another project, but tailored to this repository's actual domain, docs, codebase, and conventions.

## Objective

Implement a durable AI-manageable knowledge base under `docs/private/llm_knowledge/`, and standardize agent instructions so that Copilot, Claude, and other agents can understand and maintain it without repeated manual guidance.

Do not stop at a plan. Implement the structure end-to-end.

## What you must do

### 1. Analyze the repository first

Before writing files, inspect the repo to understand:

- the project purpose
- the main stack and architecture
- the most important modules and concepts
- existing documentation
- whether `AGENTS.md`, `CLAUDE.md`, or other agent instruction files already exist
- whether there are architecture docs, user guides, ADRs, design notes, reports, RFCs, PRDs, or technical memos worth ingesting

Use the existing repository as the source of truth. Do not invent project facts.

### 2. Standardize agent instructions using `AGENTS.md`

Adopt this instruction hierarchy:

1. `AGENTS.md` at the repository root = repository-wide source of truth
2. `docs/private/llm_knowledge/AGENTS.md` = wiki-specific source of truth for that subtree
3. `CLAUDE.md` files = compatibility shims only, pointing to the relevant `AGENTS.md`

Important rules:

- Prefer `AGENTS.md` over `.github/copilot-instructions.md`
- Do **not** create `.github/copilot-instructions.md` unless there is a concrete existing reason in the repo to keep using it
- If `CLAUDE.md` already exists, rewrite it into a compatibility layer instead of leaving it as the main instruction source
- If `AGENTS.md` already exists, merge carefully instead of blindly overwriting it

### 3. Create the wiki architecture

Create this structure under the repo:

```text
docs/private/llm_knowledge/
├── AGENTS.md
├── CLAUDE.md
├── raw/
│   ├── project/
│   ├── articles/
│   ├── papers/
│   └── assets/
└── wiki/
    ├── index.md
    ├── log.md
    ├── overview.md
    ├── concepts/
    ├── entities/
    ├── sources/
    └── comparisons/
```

If some directories would otherwise remain empty and need to be tracked, add `.gitkeep` only where useful.

### 4. Define the wiki operating model

In `docs/private/llm_knowledge/AGENTS.md`, encode the operational rules for the wiki.

It must clearly state:

- this file governs the entire `docs/private/llm_knowledge/` subtree
- `raw/` is immutable source material
- all knowledge authoring and maintenance happens in `wiki/`
- `wiki/index.md` is the master catalog
- `wiki/log.md` is append-only history
- deeper instruction files would override higher-level ones if ever added later

Include these required conventions.

#### Required frontmatter for all wiki pages

```yaml
---
title: Human readable title
type: concept | entity | source-summary | comparison
sources:
  - raw/project/filename.md
related:
  - "[[related-page]]"
created: YYYY-MM-DD
updated: YYYY-MM-DD
confidence: high | medium | low
---
```

#### Confidence meanings

- `high`: directly supported by a `raw/` source
- `medium`: reasonable inference from multiple sources
- `low`: tentative hypothesis not yet verified

#### Naming conventions

- file names in kebab-case
- prefer English file names even if prose is in another language
- use wiki links without extension: `[[concept-name]]`
- use ISO dates `YYYY-MM-DD`
- paths in `sources:` are relative to `llm_knowledge/`

#### Required templates

Include reusable templates in the wiki `AGENTS.md` for:

1. concept page
2. entity page
3. source-summary page
4. comparison page

The concept template should have sections equivalent to:

- Definition
- How it works
- In this project
- Limits and watchpoints
- Sources

The entity template should have sections equivalent to:

- Description
- Role in the project
- Contacts / links

The source-summary template should have sections equivalent to:

- Document metadata
- Key points
- Covered concepts
- Important citations
- Notable decisions / conclusions

The comparison template should have sections equivalent to:

- Context
- Criteria table
- Analysis
- Recommendation
- Sources

### 5. Define wiki workflows

In the wiki `AGENTS.md`, define and explain these workflows:

#### INGEST

Triggered when a new source document is added to `raw/` or when the user asks for ingestion.

Required steps:

1. read the source
2. create a `wiki/sources/{slug}.md` summary page
3. update relevant existing concept/entity pages
4. create new concept/entity pages when needed
5. update `wiki/index.md`
6. append a new entry to `wiki/log.md`

The `wiki/log.md` entry format should be:

```md
## YYYY-MM-DD — INGEST : [source name]

- **Source :** `raw/[path]`
- **Pages created:** [...]
- **Pages updated:** [...]
- **Notes:** [...]
```

#### QUERY

Triggered when the user asks a question that should be answered from the wiki.

Required steps:

1. start from `wiki/index.md`
2. read the relevant pages
3. synthesize from the wiki, citing page names and `raw/` sources when useful
4. if the result is reusable knowledge, suggest saving it into the wiki

#### LINT

Triggered on explicit wiki lint/review requests.

Check for:

- orphan pages
- broken wiki links
- contradictory claims across pages
- missing dedicated pages for frequently referenced concepts
- wiki citations pointing to missing source-summary pages
- stale `updated` dates that should be reviewed

### 6. Create repository-wide agent instructions

Create or update the root `AGENTS.md` so it becomes the repository-wide source of truth.

It should include:

- scope and precedence
- short project description derived from this repo
- user/repo working preferences if they are clearly discoverable
- stack summary
- architecture summary
- critical engineering rules already evident in the repo
- contribution conventions if they exist
- a routing section explaining that any work under `docs/private/llm_knowledge/` must also follow `docs/private/llm_knowledge/AGENTS.md`

Do **not** fill it with generic filler. Make it specific to this repository.

### 7. Rewrite or create `CLAUDE.md` as compatibility shims

#### Root `CLAUDE.md`

Create or rewrite it so that it is **not** the main source of truth anymore.

It should explicitly say:

- this file exists for agents that read `CLAUDE.md`
- repository-wide source of truth is `AGENTS.md`
- for any work under `docs/private/llm_knowledge/`, also follow `docs/private/llm_knowledge/AGENTS.md`

It may also keep a short project snapshot and a few critical reminders, but avoid duplicating the full repo instructions.

#### Wiki `docs/private/llm_knowledge/CLAUDE.md`

Create or rewrite it as a minimal compatibility shim that says:

- this file exists for agents that read `CLAUDE.md`
- source of truth for this subtree is `AGENTS.md`
- never modify `raw/`
- wiki maintenance happens under `wiki/`
- keep `wiki/index.md` and `wiki/log.md` aligned with ingest work

### 8. Initialize the wiki with real repository knowledge

Do not leave the wiki as empty scaffolding.

Populate it with an initial useful knowledge base derived from this repository.

#### Source ingestion

Identify the most relevant existing internal docs, such as:

- `README.md`
- developer guides
- architecture docs
- user guides
- ADRs
- RFCs
- technical notes
- reports
- design docs
- security docs
- deployment docs
- integration docs

Copy the selected original source documents into `docs/private/llm_knowledge/raw/project/` as immutable copies.

Rules:

- do not modify the original docs in place
- do not "summarize into raw"; raw should be source copies
- preserve filenames where practical, adapting only if necessary for clarity/consistency

#### Initial wiki pages

Create:

- `wiki/overview.md`
- source-summary pages for the ingested raw sources
- a first set of concept pages representing the project's core technical model
- relevant entity pages for key organizations, systems, or actors if the repo contains enough evidence
- comparison pages only if the repo already contains comparative decision material

Use judgment: create enough pages to make the wiki genuinely useful, but do not generate hollow placeholders.

### 9. Build `wiki/index.md`

Create or update `wiki/index.md` as the master catalog.

It should contain, at minimum:

- overview
- concepts
- entities
- ingested sources
- comparisons

Use grouped sections and short descriptions for each page.

### 10. Build `wiki/log.md`

Create or update `wiki/log.md` as append-only history.

Add an initialization entry documenting:

- creation of the wiki structure
- instruction files created or updated
- source files ingested into `raw/project/`
- wiki pages created
- major notes or limits discovered during initialization

Do not delete previous entries if the file already exists.

### 11. Preserve repo reality

Important constraints:

- do not invent project architecture, APIs, business entities, or operational history
- do not create fake citations
- if something is uncertain, mark it with appropriate confidence
- if the repo already has a knowledge structure, extend and harmonize it rather than replacing it blindly
- do not duplicate the same instructions across multiple files unless needed for compatibility
- keep `AGENTS.md` as the normative layer and `CLAUDE.md` as compatibility only

### 12. Validation before finishing

Before concluding, verify:

- `AGENTS.md` exists at the top level
- `docs/private/llm_knowledge/AGENTS.md` exists
- both `CLAUDE.md` files point to the new `AGENTS.md` hierarchy
- `docs/private/llm_knowledge/raw/` exists and contains real source material
- `wiki/index.md`, `wiki/log.md`, and `wiki/overview.md` exist
- at least one real source-summary page exists
- at least a few real concept pages exist
- the wiki instructions are specific enough that another agent could maintain the wiki later without further explanation

### 13. Final response

When done, respond with a concise implementation summary that includes:

- which instruction files were created or rewritten
- which raw sources were ingested
- which initial wiki pages were created
- any repo-specific limitations or follow-up gaps you found

## Style and execution requirements

- Be autonomous and implement the work fully.
- Make precise, surgical changes.
- Prefer repository-specific content over generic boilerplate.
- Keep prose concise but complete.
- If documentation already exists, reuse and structure it instead of rewriting everything from scratch.
- Do not ask for permission unless you hit a real ambiguity that materially changes the architecture.
