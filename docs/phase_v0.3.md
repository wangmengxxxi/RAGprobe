# RAGProbe v0.3 - Testset Generation

## Goal

v0.3 starts reducing the cost of building and maintaining RAG retrieval testsets.

The phase focuses on a lightweight, deterministic testset maintenance loop:

```text
chunks.jsonl -> generated testset -> manual bad case additions -> review sample
```

This phase intentionally does not implement full LLM QA generation, PDF parsing,
or embedding-based hard negative mining. Those remain future enhancements.

## Implemented Scope

- Chunk input support:
  - JSONL
  - JSON list
  - JSON object with `chunks`
  - JSON object mapping `chunk_id` to content
- Lightweight testset generation:
  - `ragprobe generate --chunks chunks.jsonl --output testset.json`
  - deterministic query generation from chunk metadata/content
  - `metadata.chunks` included in generated testsets
- Lightweight hard negative mining:
  - text overlap similarity
  - metadata difference detection
  - initial confusion type inference:
    - `subject_confusion`
    - `condition_confusion`
    - `event_confusion`
    - `temporal_confusion`
    - `scope_confusion`
    - `semantic_only`
- Manual bad case maintenance:
  - `ragprobe add-case`
  - converts real production failures into regression test cases
- Review sampling:
  - `ragprobe sample`
  - exports JSONL rows with a `review` field for human acceptance notes
- Validation enhancement:
  - warns when expected chunks or hard negatives are not present in `metadata.chunks`

## User Value

Users no longer need to start from a blank testset.

They can:

1. Generate a first-pass testset from existing chunks.
2. Validate it.
3. Add real failure cases from production.
4. Sample cases for human review.
5. Reuse the same testset in `run`, `diagnose`, `compare`, and `check`.

## Functional Test

```bat
python -m pytest
python -m ragprobe generate ^
  --chunks tests\fixtures\chunks.jsonl ^
  --output tests\tmp_outputs\v03_generated.json ^
  --num-cases 3 ^
  --name v03-smoke
python -m ragprobe validate ^
  --testset tests\tmp_outputs\v03_generated.json
python -m ragprobe add-case ^
  --testset tests\tmp_outputs\v03_generated.json ^
  --output tests\tmp_outputs\v03_added.json ^
  --query "买方付款通知需要多久确认？" ^
  --expected-chunk buyer_payment_notice ^
  --hard-negative buyer_payment_30 ^
  --confusion-type event_confusion ^
  --tag manual
python -m ragprobe sample ^
  --testset tests\tmp_outputs\v03_added.json ^
  --output tests\tmp_outputs\v03_review.jsonl ^
  --limit 3
type tests\tmp_outputs\v03_review.jsonl
```

## Success Criteria

- Generated testset contains cases, expected chunks, hard negatives, and `metadata.chunks`.
- Generated testset passes `ragprobe validate`.
- Manual bad cases can be appended and reused as regression cases.
- Review sample output is JSONL and includes `review.accepted` and `review.notes`.

## Next Step

The natural v0.4 direction is smarter diagnosis and stronger generation:

- better hard negative mining with embeddings
- optional LLM QA generation
- stronger failure classification
- clearer evidence-backed recommendations
- CI/GitHub Actions polish
