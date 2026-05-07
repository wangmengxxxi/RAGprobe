# RAGProbe v0.1 - Diagnostic Core

## Goal

v0.1 proves the core diagnostic loop:

```text
manual testset + offline retrieval results -> diagnostic metrics -> failure report
```

This phase does not require an LLM key and does not call a user's live retriever.

## Implemented Scope

- Core data models: `TestSet`, `TestCase`, `RetrievalResult`, `DiagnosticReport`.
- Offline JSON loading for testsets and retrieval results.
- Deterministic retrieval metrics:
  - hit rate
  - MRR
  - precision@5 / precision@10
  - hard negative FPR
  - confusion distribution
- Failure case extraction:
  - miss
  - false positive
  - both
- Report renderers:
  - terminal
  - Markdown
  - JSON
- CLI commands:
  - `ragprobe demo`
  - `ragprobe diagnose`
  - `ragprobe compare`
  - `ragprobe check`

## User Value

Users can run RAGProbe on prepared retrieval results and see where retrieval fails,
especially whether hard negatives are being retrieved together with, or instead of,
the correct chunks.

## Functional Test

```bat
python -m pytest
python -m ragprobe demo
python -m ragprobe diagnose ^
  --testset tests\fixtures\contract_testset.json ^
  --results tests\fixtures\results_v1.json
python -m ragprobe compare ^
  --testset tests\fixtures\contract_testset.json ^
  --before tests\fixtures\results_v1.json ^
  --after tests\fixtures\results_v2.json
python -m ragprobe check ^
  --testset tests\fixtures\contract_testset.json ^
  --results tests\fixtures\results_v1.json ^
  --min-hit-rate 0.7 ^
  --max-fpr 0.3
```

## Success Criteria

- Demo runs without external services.
- Diagnostic report includes metrics, confusion distribution, worst cases, and recommendations.
- Compare report shows metric deltas and improved/regressed cases.
- Check command returns a non-zero exit code when thresholds fail.
