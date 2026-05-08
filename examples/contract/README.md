# Contract Example

This directory contains a tiny contract retrieval setup that exercises the three
RAGProbe integration paths.

## Python Retriever

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever examples/contract/python_retriever.py \
  --output .tmp/contract-python-results.json

python -m ragprobe diagnose \
  --testset examples/contract/testset.json \
  --results .tmp/contract-python-results.json \
  --format markdown \
  --output .tmp/contract-python-report.md
```

## Node.js JSONL Retriever

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever-cmd "node examples/contract/node_jsonl_retriever.js" \
  --output .tmp/contract-node-results.json
```

RAGProbe sends one JSON object per line to stdin:

```json
{"query":"买方逾期付款超过30天的违约金是多少？","top_k":10}
```

The process writes one JSON array per line to stdout:

```json
[{"chunk_id":"buyer_payment_30","content":"...","score":0.95}]
```

## HTTP Endpoint

Start the example server in one terminal:

```bash
python examples/contract/http_retriever_server.py
```

Run RAGProbe in another terminal:

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --endpoint http://127.0.0.1:8008/search \
  --endpoint-config examples/contract/endpoint_config.json \
  --output .tmp/contract-http-results.json
```

## Built-in Baseline

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --baseline embedding \
  --output .tmp/contract-embedding-baseline-results.json
```

## CI Check

```bash
python -m ragprobe check \
  --testset examples/contract/testset.json \
  --results .tmp/contract-python-results.json \
  --min-hit-rate 0.7 \
  --min-mrr 0.5 \
  --max-fpr 1.0
```

## Compare Against a Weak Retriever

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever examples/contract/weak_python_retriever.py \
  --output .tmp/contract-weak-results.json

python -m ragprobe compare \
  --testset examples/contract/testset.json \
  --before .tmp/contract-weak-results.json \
  --after .tmp/contract-python-results.json
```

## Multi-Retriever Experiment

```bash
python -m ragprobe experiment \
  --config examples/contract/experiment.json \
  --output-dir .tmp/contract-experiment
```

## Audit and Repair

Audit a small sample with an LLM judge:

```bash
python -m ragprobe audit \
  --testset examples/contract/testset.json \
  --output .tmp/contract-audit.json \
  --markdown .tmp/contract-audit.md \
  --llm qwen \
  --model qwen-plus \
  --sample-size 1
```

Create a reviewable repair plan:

```bash
python -m ragprobe repair-plan \
  --audit-report .tmp/contract-audit.json \
  --output .tmp/contract-repair-plan.json \
  --markdown .tmp/contract-repair-plan.md
```

Apply safe fixes to a new testset file:

```bash
python -m ragprobe apply-audit-fixes \
  --testset examples/contract/testset.json \
  --repair-plan .tmp/contract-repair-plan.json \
  --output .tmp/contract-fixed-testset.json \
  --report .tmp/contract-repair-apply.md
```
