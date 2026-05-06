# Qualira

Phase 0 proves the core question from `QUALIRA_DESIGN.md`:

> Can `EvidenceUnit + FieldClaim + Boundary` reduce hard-negative false positives compared with naive similarity top-k?

This first slice intentionally uses manual contract fixtures only. It does not include LLM extraction, PDF parsing, vector search, Web UI, adapters, or automatic ingestion.

## Run

Use the conda environment named `qualira`.

```powershell
conda run -n qualira python -m qualira.cli.ingest tests/fixtures/contract_evidence.json --db qualira.db --reset
conda run -n qualira python -m qualira.cli.query "买方逾期付款超过30天有什么责任？" --db qualira.db --explain
conda run -n qualira python -m qualira.cli.eval
```

Expected Phase 0 benchmark shape:

```text
| Method | EHR | FPR | BER | SGR |
| --- | --- | --- | --- | --- |
| naive_chunk_topk | high | high | N/A | N/A |
| qualira | high | much lower | measurable | 1.0000 |
```

The fixture includes 20 contract queries and hard negatives covering subject swaps, event swaps, condition mismatches, answer type mismatches, and scope mismatches.
