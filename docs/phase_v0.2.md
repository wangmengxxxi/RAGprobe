# RAGProbe v0.2 - Easy Integration

## Goal

v0.2 lets users connect their own retriever without redesigning their RAG system.

The phase expands RAGProbe from offline result diagnosis to live retrieval execution:

```text
testset -> user retriever -> results.json -> diagnose / compare / check
```

## Implemented Scope

- Python retriever script integration:
  - `ragprobe run --retriever retriever.py`
  - same-process `importlib` loading
  - requires RAGProbe and retriever dependencies in the same Python environment
- Cross-language subprocess integration:
  - `ragprobe run --retriever-cmd "node retriever.js"`
  - JSONL stdin/stdout protocol
  - suitable for Node.js, Java, Go, C#, Rust, and other languages
- HTTP endpoint integration:
  - `ragprobe run --endpoint URL`
  - endpoint config supports headers, timeout, and batch size
- Offline bridge commands:
  - `ragprobe export-queries`
  - `ragprobe import-results`
- Schema validation:
  - `ragprobe validate --testset`
  - `ragprobe validate --testset --results`
- Matching improvements:
  - prefer exact `chunk_id`
  - fallback to content matching when chunk IDs are missing
  - report matching confidence and matching notes
- Score handling:
  - `score` is optional
  - RAGProbe respects retriever result order
  - RAGProbe does not compare absolute score values across retrievers

## User Value

Python users can expose `retrieve(query, top_k)`.

Non-Python users can use the JSONL subprocess protocol instead of wrapping their
retriever in an HTTP service.

Teams with existing offline QA data can export queries and import results without
hand-writing the full RAGProbe result schema.

## Functional Test

```bat
python -m pytest
python -m ragprobe run ^
  --testset tests\fixtures\contract_testset.json ^
  --retriever tests\fixtures\retriever_with_ids.py ^
  --output tests\tmp_outputs\v02_run_ids.json
python -m ragprobe run ^
  --testset tests\fixtures\contract_testset.json ^
  --retriever-cmd "python tests\fixtures\jsonl_retriever.py" ^
  --output tests\tmp_outputs\v02_run_cmd.json
python -m ragprobe export-queries ^
  --testset tests\fixtures\contract_testset.json ^
  --output tests\tmp_outputs\v02_queries.jsonl
python -m ragprobe import-results ^
  --queries tests\tmp_outputs\v02_queries.jsonl ^
  --results tests\fixtures\import_results.jsonl ^
  --output tests\tmp_outputs\v02_imported.json
python -m ragprobe validate ^
  --testset tests\fixtures\contract_testset.json ^
  --results tests\fixtures\results_without_ids.json
```

## Success Criteria

- Python retriever, JSONL subprocess retriever, and offline import all produce valid results.
- Missing `chunk_id` cases can still be diagnosed through content fallback.
- Validation warns about low-confidence input rather than silently producing misleading output.
