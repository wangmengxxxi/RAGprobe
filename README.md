# RAGProbe

> Diagnose your RAG before your users do.

RAGProbe is a retrieval diagnostics and regression-testing CLI for RAG systems.
It focuses on the retrieval layer: can your system find the right chunks, avoid
similar-but-wrong chunks, and keep that behavior from regressing in CI?

## Why It Exists

Your contract RAG system receives this query:

```text
买方逾期付款超过30天的违约金是多少？
```

But the retriever returns this chunk near the top:

```text
卖方延期交货超过15日，应承担延期交货违约责任。
```

It looks similar, but the subject, event, and condition are wrong. RAGProbe is
designed to catch this retrieval failure before users build an answer on it.

## What It Measures

RAGProbe runs a hard-negative testset against your retriever and reports:

- retrieval quality: hit rate, MRR, precision@k
- hard-negative resistance: FPR and confusion distribution
- failure patterns: misses, ranking weakness, hard negatives above correct chunks
- system issue signals and evidence-backed recommendations
- CI-friendly threshold checks

The core diagnostic loop does not require an LLM key.

```text
testset -> run retriever -> diagnose -> compare/check -> improve testset quality
```

## Install

From a local checkout:

```bash
python -m pip install -e ".[dev]"
```

After package release, regular installation will be:

```bash
pip install ragprobe
```

## Five-Minute Start

Run the bundled offline demo:

```bash
python -m ragprobe demo
```

Run the contract example against a Python retriever:

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever examples/contract/python_retriever.py \
  --output .tmp/contract-results.json

python -m ragprobe diagnose \
  --testset examples/contract/testset.json \
  --results .tmp/contract-results.json
```

Run a built-in local baseline without writing a retriever script:

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --baseline embedding \
  --output .tmp/embedding-baseline-results.json
```

Write a Markdown report:

```bash
python -m ragprobe diagnose \
  --testset examples/contract/testset.json \
  --results .tmp/contract-results.json \
  --format markdown \
  --output .tmp/contract-report.md
```

## Generate a Testset From Chunks

RAGProbe can create a deterministic starter testset from your existing chunks.
This is intended as a cold-start scaffold and CI baseline.

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/generated-testset.json \
  --hard-negative-top-k 2 \
  --hn-strategy hybrid \
  --quality-report .tmp/generated-quality.md

python -m ragprobe validate --testset .tmp/generated-testset.json
```

Generated cases include quality metadata and warnings so you can decide whether
they are ready to become regression tests. Add real production bad cases over
time:

```bash
python -m ragprobe add-case \
  --testset .tmp/generated-testset.json \
  --output .tmp/generated-testset-with-bad-case.json \
  --query "买方逾期付款的责任是什么？" \
  --expected-chunk buyer_payment_30 \
  --hard-negative seller_delivery_15 \
  --confusion-type subject_confusion
```

## Optional LLM-Assisted Generation

The default generation path is deterministic and does not call any model. For
higher-quality query phrasing and hard-negative judgment, v0.7 adds opt-in
generation through OpenAI-compatible chat completion APIs.

Set your API key in the environment:

```bash
export AI_API_KEY="..."
```

Then run:

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/llm-testset.json \
  --llm openai-compatible \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  --model qwen-plus \
  --yes \
  --quality-report .tmp/llm-quality.md
```

Qwen has a preset, so this shorter command is equivalent for DashScope:

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/qwen-testset.json \
  --llm qwen \
  --model qwen-plus \
  --yes
```

Notes:

- RAGProbe reads API keys from an environment variable, defaulting to `AI_API_KEY`.
- Use `--api-key-env NAME` if a provider key lives in a different variable.
- Never write API keys into testsets, caches, or commits.
- LLM generation uses `.ragprobe_cache/` by default to avoid repeated calls.
- `.ragprobe_cache/` and local output directories are ignored by git.
- `diagnose`, `compare`, and `check` remain zero-LLM deterministic commands.

For stricter generation-time validation, add an LLM judge pass:

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/llm-validated-testset.json \
  --llm openai-compatible \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  --model qwen-plus \
  --llm-validate \
  --yes
```

The validation pass asks whether the expected chunk can answer the generated
query and whether each hard negative can also answer it. Answerable hard
negatives are removed; cases whose expected chunk cannot answer the query are
rejected unless `--keep-rejected` is passed.

## Python API

You can use the same workflow from Python code without shelling out to the CLI:

```python
from ragprobe import RAGProbe

probe = RAGProbe()

testset = probe.generate(
    chunks="examples/contract/chunks.jsonl",
    hard_negative_top_k=2,
)

results = probe.run(
    testset=testset,
    retriever="examples/contract/python_retriever.py",
)

report = probe.diagnose(testset=testset, results=results)
check = probe.check(report, min_hit_rate=0.7, min_mrr=0.5, max_fpr=0.3)

print(report.hit_rate, report.mrr, report.fpr)
print(check.passed)
```

For OpenAI-compatible providers, pass `llm`, `base_url`, and `model`:

```python
from ragprobe import RAGProbe

probe = RAGProbe(
    llm="openai-compatible",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    model="qwen-plus",
)

report = probe.evaluate(
    chunks="examples/contract/chunks.jsonl",
    retriever="examples/contract/python_retriever.py",
    hard_negative_top_k=2,
    llm_validate=True,
)

print(report.hit_rate, report.fpr)
```

For Qwen, the shorter preset also works:

```python
probe = RAGProbe(llm="qwen", model="qwen-plus")
```

Multi-retriever experiment:

```python
report = probe.experiment(
    config="examples/contract/experiment.json",
    output_dir=".tmp/contract-experiment",
)
```

Built-in baseline retriever:

```python
results = probe.run(
    testset="examples/contract/testset.json",
    baseline="embedding",
    top_k=10,
)
```

Audit and repair:

```python
audit = probe.audit(
    testset="examples/contract/testset.json",
    output=".tmp/audit.json",
    markdown=".tmp/audit.md",
    sample_size=1,
)

plan = probe.repair_plan(
    audit_report=audit,
    output=".tmp/repair-plan.json",
)

result = probe.apply_audit_fixes(
    testset="examples/contract/testset.json",
    repair_plan=plan,
    output=".tmp/fixed-testset.json",
)
```

## Retriever Integration

### Python

Provide a file exposing `retrieve(query, top_k)`:

```python
def retrieve(query: str, top_k: int = 10) -> list[dict]:
    return [
        {
            "chunk_id": "buyer_payment_30",
            "content": "买方逾期付款超过30天，应按未付款金额支付违约金。",
            "score": 0.95,
            "metadata": {},
        }
    ][:top_k]
```

Run it:

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever examples/contract/python_retriever.py \
  --output .tmp/python-results.json
```

### Node.js or Any JSONL Process

RAGProbe can talk to any subprocess over stdin/stdout JSONL:

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever-cmd "node examples/contract/node_jsonl_retriever.js" \
  --output .tmp/node-results.json
```

### HTTP Endpoint

Start the example server:

```bash
python examples/contract/http_retriever_server.py
```

Run against it:

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --endpoint http://127.0.0.1:8008/search \
  --endpoint-config examples/contract/endpoint_config.json \
  --output .tmp/http-results.json
```

### Built-in Baselines

RAGProbe includes deterministic local baselines that search
`testset.metadata.chunks` directly:

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --baseline embedding \
  --output .tmp/embedding-baseline-results.json
```

Supported baseline names:

- `lexical`: token-overlap scoring
- `embedding`: hashed token-vector cosine scoring

These baselines do not call an API, download a model, or require an LLM key. They
are intended as reproducible comparison anchors for CI and experiments, not as a
replacement for your production embedding stack.

## Compare Retriever Changes

Use the same testset before and after changing chunking, embedding, reranking, or
filters:

```bash
python -m ragprobe compare \
  --testset examples/contract/testset.json \
  --before .tmp/contract-weak-results.json \
  --after .tmp/contract-results.json
```

The comparison reports metric deltas and improved/regressed cases.

## Multi-Retriever Experiments

Run several named retrievers from one JSON config:

```bash
python -m ragprobe experiment \
  --config examples/contract/experiment.json \
  --output-dir .tmp/contract-experiment
```

This writes per-retriever results and reports plus:

```text
.tmp/contract-experiment/experiment_report.json
.tmp/contract-experiment/experiment_report.md
```

## Audit and Repair Testsets

The core diagnostic commands stay deterministic and zero-LLM. Testset audit is
an optional LLM workflow for checking whether generated or manually maintained
cases are trustworthy.

Audit a small sample with Qwen:

```bash
python -m ragprobe audit \
  --testset examples/contract/testset.json \
  --output .tmp/audit.json \
  --markdown .tmp/audit.md \
  --llm qwen \
  --model qwen-plus \
  --sample-size 1
```

Build a reviewable repair plan:

```bash
python -m ragprobe repair-plan \
  --audit-report .tmp/audit.json \
  --output .tmp/repair-plan.json \
  --markdown .tmp/repair-plan.md
```

Apply safe fixes to a new testset file:

```bash
python -m ragprobe apply-audit-fixes \
  --testset examples/contract/testset.json \
  --repair-plan .tmp/repair-plan.json \
  --output .tmp/fixed-testset.json \
  --report .tmp/repair-apply.md
```

`reject_case` actions are skipped by default. To remove failed cases from the
output testset, pass `--allow-reject-cases` explicitly.

## CI Usage

Use `check` to fail a build when retrieval quality crosses thresholds:

```bash
python -m ragprobe check \
  --testset examples/contract/testset.json \
  --results .tmp/contract-results.json \
  --min-hit-rate 0.7 \
  --min-mrr 0.5 \
  --max-fpr 0.3
```

See [.github/workflows/ragprobe.yml](.github/workflows/ragprobe.yml) for a
copyable GitHub Actions example.

## File Formats

RAGProbe v1.0 writes stable schema metadata into JSON artifacts:

```json
{
  "metadata": {
    "schema_version": "ragprobe.testset.v1",
    "ragprobe_version": "1.0.0"
  }
}
```

The committed contract example shows all core formats:

- chunks JSONL: [examples/contract/chunks.jsonl](examples/contract/chunks.jsonl)
- testset JSON: [examples/contract/testset.json](examples/contract/testset.json)
- Python retriever: [examples/contract/python_retriever.py](examples/contract/python_retriever.py)
- JSONL subprocess retriever: [examples/contract/node_jsonl_retriever.js](examples/contract/node_jsonl_retriever.js)
- HTTP endpoint retriever: [examples/contract/http_retriever_server.py](examples/contract/http_retriever_server.py)
- endpoint config: [examples/contract/endpoint_config.json](examples/contract/endpoint_config.json)

## Local Verification

```bash
python -m pytest
python -m ruff check src tests
python -m ragprobe --version
python -m ragprobe demo
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/generated-testset.json \
  --quality-report .tmp/generated-quality.md
python -m ragprobe validate --testset .tmp/generated-testset.json
```

## What RAGProbe Does Not Do

- It does not evaluate final LLM answer quality.
- It is not a RAG framework or vector database.
- It is not a real-time monitoring dashboard.
- It does not require LLM scoring for the core diagnostic and CI loop.

## Roadmap

RAGProbe v1.x keeps the core JSON artifact contracts stable while adding
optional baseline, experiment, and release-polish features around them.

## License

MIT
