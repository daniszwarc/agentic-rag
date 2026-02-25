# agentic-rag

A Corrective RAG (CRAG) implementation using LangGraph. The system routes questions intelligently between a local vector store and web search, grades retrieved documents for relevance, and validates generated answers for hallucinations before returning a response.

This was built as a learning project to explore agentic RAG patterns — specifically how to make retrieval self-correcting rather than a single-pass lookup.

---

## What it does

Standard RAG retrieves documents and generates an answer in one pass, with no feedback loop. This system adds three grading steps:

1. **Document relevance grading** — retrieved chunks are scored individually; irrelevant ones are discarded and trigger a web search fallback
2. **Hallucination grading** — the generated answer is checked against the retrieved documents before being returned
3. **Answer grading** — the answer is checked against the original question to confirm it actually resolves it

If the answer fails either post-generation check, the graph re-routes to web search or regenerates rather than returning a bad response.

---

## Graph architecture

```
            Question
               │
               ▼
    ┌─────────────────────┐
    │    Route Question   │  Decides: vectorstore or web search?
    └──────────┬──────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   Retrieve         Web Search
   (Chroma)         (Tavily)
       │                │
       └───────┬─────────┘
               ▼
    ┌─────────────────────┐
    │   Grade Documents   │  Each doc graded: relevant or not?
    └──────────┬──────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
   Generate         Web Search  ← fallback if docs insufficient
       │
       ▼
    ┌─────────────────────┐
    │ Hallucination Check │  Is answer grounded in documents?
    └──────────┬──────────┘
               │
       ┌───────┼────────────┐
       ▼       ▼            ▼
    useful  not useful  not supported
       │       │            │
      END   Web Search   Regenerate
```

The compiled graph is exported as `graph.png` at startup.

---

## Project structure

```
agentic-rag/
├── main.py                         Entry point
├── ingestion.py                    Document loading, chunking, Chroma ingestion
├── graph/
│   ├── graph.py                    LangGraph StateGraph definition
│   ├── state.py                    GraphState TypedDict
│   ├── consts.py                   Node name constants
│   ├── nodes/
│   │   ├── retrieve.py             Chroma retrieval node
│   │   ├── grade_documents.py      Document relevance grading node
│   │   ├── generate.py             Answer generation node
│   │   └── web_search.py           Tavily web search fallback node
│   └── chains/
│       ├── router.py               Question router (vectorstore vs web search)
│       ├── retrieval_grader.py     Document relevance grader
│       ├── generation.py           RAG generation chain
│       ├── hallucination_grader.py Hallucination checker
│       └── answer_grader.py        Answer quality checker
└── graph.png                       Auto-generated graph diagram
```

---

## Knowledge base

The vector store is seeded with three articles from Lilian Weng's blog, covering:

- [LLM-powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)

The router is configured to send questions on these topics to the vector store, and everything else to web search.

Documents are chunked at 250 tokens with no overlap using `RecursiveCharacterTextSplitter` with tiktoken encoding, then stored in a local Chroma collection (`rag-chroma`).

---

## Grading chains

All graders use structured output with Pydantic models and `binary_score` fields, which keeps conditional routing deterministic — no string parsing.

| Chain | Model | Purpose |
|---|---|---|
| `router` | GPT-4o | Route question to vectorstore or web |
| `retrieval_grader` | GPT-4o | Score each retrieved doc: relevant or not |
| `generation_chain` | GPT-4o | Generate answer from context (rlm/rag-prompt) |
| `hallucination_grader` | GPT-4o | Check if answer is grounded in documents |
| `answer_grader` | GPT-4o | Check if answer resolves the question |

---

## Setup

**Requirements:** Python 3.13+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/daniszwarc/agentic-rag.git
cd agentic-rag
uv sync
```

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

Ingest documents into Chroma (first run only — uncomment the `Chroma.from_documents` block in `ingestion.py`):

```bash
uv run python ingestion.py
```

Run the agent:

```bash
uv run python main.py
```

---

## State

The graph passes state through all nodes as a typed dict:

```python
class GraphState(TypedDict):
    question:   str               # original user question (last-write wins)
    generation: str               # current LLM answer
    web_search: bool              # flag: should we run web search?
    documents:  Annotated[List[str], operator.add]  # accumulated documents
```

`documents` uses `operator.add` so web search results append to retrieved docs rather than replacing them.

---

## Dependencies

Managed with [uv](https://github.com/astral-sh/uv). Key packages:

- `langgraph` — graph orchestration
- `langchain-openai` — LLM and embeddings
- `langchain-chroma` — local vector store
- `langchain-tavily` — web search fallback
- `langchainhub` — pulls the `rlm/rag-prompt` generation prompt
