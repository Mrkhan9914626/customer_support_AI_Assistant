# TechFlow Solutions — RAG Project

This repository contains a Retrieval-Augmented Generation (RAG) assistant for TechFlow Solutions. The core functionality is implemented in `database.py`, which provides a set of search and answer tools used by the agent in `app.py`.

This README documents each tool exported from `database.py`: `semantic_search`, `keyword_search`, `get_document_section`, `hybrid_search`, and `answer_with_context`. For each tool we describe the purpose, inputs/outputs, how it works, example usage, and common edge-cases / notes.

---

## Table of Contents

- semantic_search
- keyword_search
- get_document_section
- hybrid_search
- answer_with_context

---

## 1) semantic_search

Purpose
- Perform semantic similarity search over the ingested document corpus. Use this when you need conceptually related passages rather than exact phrase matches (for example: "how to integrate with Slack", "what does the pricing tier include").

Inputs
- `query` (str): Natural language question or search phrase.
- `num_results` (int, optional): Number of results to return (default 5, capped to 10).

Outputs
- JSON string containing:
  - `status`: "success" or "error"
  - `num_results`: number of returned results
  - `average_relevance`: average calculated relevance score across results
  - `query_categories`: list of identified categories (see `identify_query_category` in code)
  - `results`: array of result objects each with `text`, `metadata`, `similarity_score`, and `relevance_score`
  - `suggestion`: optional advice when relevance is low

How it works
- The function calls the vector store's `similarity_search_with_score` (Pinecone-backed) which finds the top-k vectors closest to the query embedding.
- Each returned document chunk is paired with a similarity score from the vector store.
- The code formats those results, computes a simple `relevance_score` (keyword overlap or a custom formula) and returns structured JSON for the agent to consume.

When to use
- Best for open-ended or conceptual queries.

Example
- `semantic_search("How do I configure SSO for FlowDesk?", num_results=5)`

Edge cases and notes
- Requires embeddings and vector index to be initialized and contain documents.
- If the vector index is empty or the API key is missing, function will return an error JSON.
- Similarity thresholds depend on the embedding model and index; tune `num_results` and rerank thresholds as needed.

---

## 2) keyword_search

Purpose
- Perform exact keyword/phrase search over the raw document text. Use this for highly factual queries where literal terms or phrases are important (for example: "pricing", "support email", specific product names).

Inputs
- `query` (str): Keyword phrase or query text.
- `num_results` (int, optional): Number of results to return (default 5).

Outputs
- JSON string with `status`, `num_results`, `results` array containing `text`, `match_score` and `metadata`, and optional `suggestion`.

How it works
- The implementation searches the `ALL_DOCUMENTS` list (which is filled with chunked `page_content` values produced by the text splitter).
- It computes a simple `match_score` such as term frequency, phrase match, or overlap between words.
- Matches are sorted by `match_score` and the top results returned.

When to use
- When searching for exact words, contact details, policy lines, code snippets, or specific terms.

Example
- `keyword_search("refund policy", num_results=3)`

Edge cases and notes
- This method uses string-based matching; it won't find paraphrases (use `semantic_search` instead).
- Very short queries can return many noisy matches; consider requiring a minimum query length or exact-phrase detection.
- Indexing: `ALL_DOCUMENTS` must be populated from the text-splitter output.

---

## 3) get_document_section

Purpose
- Return aggregated content for a named document section/category (e.g., pricing, features, support, faq, policies, technical, troubleshooting).

Inputs
- `section_name` (str): One of the known section keys (pricing, features, support, faq, policies, technical, troubleshooting).

Outputs
- JSON string containing `status`, `section_name`, and `content` (one or multiple passages) or an error message.

How it works
- The function uses the `DOCUMENT_SECTIONS` mapping to identify keywords associated with each section.
- It finds document chunks from `ALL_DOCUMENTS` that contain those keywords and aggregates them into a response.
- Optionally it may prioritize higher relevance matches or deduplicate overlapping chunks.

When to use
- When the user's intent is categorical: "Show me pricing details" or "Show API integration details".

Example
- `get_document_section("pricing")`

Edge cases and notes
- Section extraction is keyword-based — it can miss content that doesn't use canonical keywords.
- Consider adding more keyword synonyms or improving section detection using a classifier.

---

## 4) hybrid_search

Purpose
- Combine semantic and keyword search to provide comprehensive results. This is useful for complex queries where both exact mentions and conceptual matches matter (for example: "compare enterprise vs business plan pricing and features").

Inputs
- `query` (str): The query text.
- `num_results` (int, optional): Number of results per method to combine.

Outputs
- JSON string with combined `semantic_results`, `keyword_results`, merged `results`, scores, and a `suggestion` field.

How it works
- Run `semantic_search` and `keyword_search` internally to produce two ranked lists.
- Merge lists, deduplicate by chunk text or metadata, and re-rank by a combined score (weighted sum of similarity + match_score + relevance).
- Return a structured JSON that contains both the separate outputs and the merged answer for the agent to use.

When to use
- Best for multi-faceted or high-importance queries where recall matters.

Example
- `hybrid_search("enterprise pricing vs business plan features", num_results=5)`

Edge cases and notes
- Combining scores from different systems requires normalization; adjust weights to balance recall vs precision.
- Watch for duplicate passages where both methods return the same chunk; deduplication is recommended.

---

## 5) answer_with_context

Purpose
- Synthesize a human-readable answer using retrieved context passages. Called after search tools gather relevant passages.

Inputs
- `query` (str): The user's question.
- `context` (str): Concatenated passages or JSON results from searches.

Outputs
- A generated answer string (usually produced by the configured LLM).

How it works
- The function prepares a prompt (via `ChatPromptTemplate`) which includes the user's query and the relevant context passages.
- It calls the configured LLM (`ChatGoogleGenerativeAI` in this project) with that prompt and returns the model output.

When to use
- Final step: after collecting 3–5 relevant passages, call this to form a concise, sourced answer.

Example
- `answer_with_context(query, context)` where `context` is the top passages returned by `hybrid_search`.

Edge cases and notes
- Ensure the context size fits LLM input limits (truncate/score passages before passing).
- Include citation metadata when returning facts: e.g., include chunk index or page number so the agent can show a source.
- If the LLM key or model is misconfigured, the function will raise an error or return an error JSON.


## How to Run
1. Install dependencies:
2. Set up environment variables for API keys.


