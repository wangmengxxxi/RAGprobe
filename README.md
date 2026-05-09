<p align="center">
  <img src="assets/brand/logo.svg" alt="RAGProbe logo" width="720">
</p>

<p align="center">
  <a href="https://pypi.org/project/ragprobe-diagnostics/"><img alt="Release" src="https://img.shields.io/badge/release-v1.5.0-blue.svg"></a>
  <a href="https://github.com/wangmengxxxi/RAGprobe/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="Core loop" src="https://img.shields.io/badge/core-zero--LLM-67e8f9.svg">
</p>

# RAGProbe

> 在用户发现 RAG 检索问题之前，先让 RAGProbe 把问题找出来。

RAGProbe 是一个面向 RAG 系统的**检索层诊断与回归测试工具**。它不评估最终
LLM 答案写得好不好，而是专注回答一个更靠前也更关键的问题：

```text
你的 retriever 是否能找对 chunk，并避开那些“看起来很像但其实错了”的 chunk？
```

RAGProbe 可以作为 CLI、Python API、CI 检查工具使用，也支持 Python、Node.js、
任意 JSONL 子进程、HTTP 服务等跨语言 retriever 接入。

## 功能

- **hard negative 抗性测试**：不只看"找没找到正确 chunk"，还显式检测"相似但错误的 chunk 是否被误召回"——这是其他 RAG 评估工具不覆盖的盲区。
- **混淆维度诊断**：自动识别 retriever 在哪个维度犯错（品牌混淆？主体混淆？时间混淆？），输出可操作的改进方向，而不只是一个笼统的分数。
- **检索方法无关**：不限于向量检索。BM25、grep、混合检索、任何能返回 chunk 的系统都能接入。
- **核心诊断零 LLM**：`run`、`diagnose`、`compare`、`check` 全部确定性执行，不需要 API key，CI 里跑不会因为 rate limit 挂掉。
- **一次生成，永久回归**：测试集生成一次后可作为回归资产反复使用。改 chunking、embedding、reranker 前后稳定对比，无额外成本。
- **跨语言接入**：Python 函数、stdin/stdout JSONL 子进程、HTTP endpoint 三种方式，Java/Go/Node.js/Rust 都能接。
- **可选 LLM 增强**：需要更自然的 query 或交叉验证时，可调用 Qwen 或 OpenAI-compatible API。不需要时完全不依赖。
- **内置 baseline 对照**：`lexical` 和本地 `embedding` baseline 开箱即用，无需外部模型即可建立对照基线。

## 解决痛点

很多开源 RAG 评估工具更关注最终答案质量，或者依赖 LLM judge 给 answer 打分。
这当然有价值，但工程落地时，retrieval 层经常先出问题：正确 chunk 没进
top-k，相似但错误的 chunk 排在正确 chunk 前面，或者一次 chunking/embedding
调整悄悄引入回归。

RAGProbe 主要补齐这些空白：

- **只看 answer 分数不够定位问题**：最终答案错了，可能是 prompt、generator、
  retriever、reranker、chunking 中任何一环的问题。RAGProbe 直接诊断 retrieval
  结果，让问题先在检索层被定位。
- **hit rate 不足以衡量 RAG 检索质量**：很多系统能召回正确 chunk，但同时也把
  高相似错误 chunk 放到前排。RAGProbe 用 hard-negative FPR 专门衡量这种风险。
- **LLM judge 成本和不稳定性不适合所有 CI 场景**：RAGProbe 的核心指标是确定性
  计算，`diagnose`、`compare`、`check` 不需要 API key，也不会受模型漂移影响。
- **跨语言系统接入门槛高**：真实 RAG 系统可能是 Java、Go、Node.js、Rust 或
  HTTP 服务。RAGProbe 支持 JSONL 子进程和 HTTP endpoint，不要求用户重写系统。
- **缺少可复用的回归资产**：RAGProbe 把测试集、hard negatives、bad cases、
  audit report、repair plan 都保存为 JSON artifact，方便长期维护和版本比较。
- **诊断报告不够可操作**：RAGProbe 不只输出分数，还输出 confusion distribution、
  failure patterns 和建议，帮助判断是 metadata filter、reranking、召回覆盖还是
  chunk 设计出了问题。


## 安装


```bash
pip install ragprobe-diagnostics
```

本地开发安装：

```bash
python -m pip install -e ".[dev]"
```

查看版本：

```bash
python -m ragprobe --version
```

## 快速开始

运行内置 demo：

```bash
python -m ragprobe demo
```

使用示例 Python retriever 跑测试集：

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever examples/contract/python_retriever.py \
  --output .tmp/contract-results.json

python -m ragprobe diagnose \
  --testset examples/contract/testset.json \
  --results .tmp/contract-results.json
```

生成 Markdown 报告：

```bash
python -m ragprobe diagnose \
  --testset examples/contract/testset.json \
  --results .tmp/contract-results.json \
  --format markdown \
  --output .tmp/contract-report.md
```

不写 retriever，直接跑内置 embedding baseline：

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --baseline embedding \
  --output .tmp/embedding-baseline-results.json
```

## 输入数据格式

### chunks.jsonl

最低要求只需要 `content`（或 `text`）字段：

```jsonl
{"content":"买方逾期付款超过30天，应按未付款金额支付违约金。"}
{"content":"卖方延期交货超过15日，应承担延期交货违约责任。"}
```

没有 `chunk_id` 时，RAGProbe 自动按顺序编号（`chunk_001`、`chunk_002`...）。
没有 `metadata` 时，仍可生成测试集和诊断指标，但 confusion type 只能推断为
`semantic_only`。

**效果更好的写法**——提供 `chunk_id` + `metadata`：

```jsonl
{"chunk_id":"c1","content":"买方逾期付款超过30天，应按未付款金额支付违约金。","metadata":{"subject":"buyer","event":"late_payment"}}
{"chunk_id":"c2","content":"卖方延期交货超过15日，应承担延期交货违约责任。","metadata":{"subject":"seller","event":"late_delivery"}}
```

各字段对诊断质量的影响：

| 字段 | 是否必填 | 不提供时的影响 |
|------|----------|----------------|
| `content`（或 `text`） | 是 | 无法生成测试集 |
| `chunk_id`（或 `id`） | 否，自动编号 | 无法跨次运行稳定追踪同一 chunk，回归对比可能不准 |
| `metadata` | 否 | confusion type 退化为 `semantic_only`，无法区分是品牌混淆、主体混淆还是时间混淆 |
| `source_document` | 否 | 报告中不显示来源文件信息 |

**推荐做法**：

- 开发初期快速验证：只提供 `content` 即可跑通
- 正式回归测试：补上 `chunk_id`，确保每次运行结果可追踪
- 精细诊断：加 `metadata`，让 RAGProbe 告诉你 retriever 具体在哪个维度犯错

带完整 metadata 的示例：

```jsonl
{"chunk_id":"p1","content":"华为手机支持66W快充。","metadata":{"brand":"华为","category":"phone"}}
{"chunk_id":"p2","content":"小米手机支持67W快充。","metadata":{"brand":"小米","category":"phone"}}
```

可能得到的 confusion type：

```text
brand_confusion
category_confusion
numeric_confusion
semantic_only
```

`url`、`page`、`id`、时间戳等非语义字段会被自动忽略，避免污染诊断标签。

### testset.json

测试集由 query、expected chunks 和 hard negatives 组成：

```json
{
  "name": "contract-demo",
  "metadata": {
    "chunks": {
      "buyer_payment_30": "买方逾期付款超过30天，应按未付款金额支付违约金。",
      "seller_delivery_15": "卖方延期交货超过15日，应承担延期交货违约责任。"
    }
  },
  "cases": [
    {
      "id": "case_1",
      "query": "买方逾期付款超过30天的违约金是多少？",
      "expected_chunks": ["buyer_payment_30"],
      "hard_negatives": [
        {
          "chunk_id": "seller_delivery_15",
          "confusion_type": "subject_confusion",
          "similarity_to_correct": 0.94,
          "reason": "同为违约责任条款，但主体和事件不同。"
        }
      ],
      "difficulty": "hard"
    }
  ]
}
```

## 生成测试集：默认确定性规则，可选 LLM

RAGProbe 的 `generate` 命令有两种路径：

- 默认路径是**确定性规则生成**，不调用任何模型，适合冷启动、CI 和没有 API key 的环境。
- 显式传入 `--llm qwen` 或 `--llm openai-compatible` 时，才会启用 LLM 辅助生成。

默认确定性生成示例：

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/generated-testset.json \
  --hard-negative-top-k 2 \
  --hn-strategy hybrid \
  --quality-report .tmp/generated-quality.md

python -m ragprobe validate --testset .tmp/generated-testset.json
```

加入真实线上 bad case。`add-case` 不依赖 LLM，适合把生产环境里真的失败过的
query 固化成长期回归测试：

```bash
python -m ragprobe add-case \
  --testset .tmp/generated-testset.json \
  --output .tmp/generated-testset-with-bad-case.json \
  --query "买方逾期付款的责任是什么？" \
  --expected-chunk buyer_payment_30 \
  --hard-negative seller_delivery_15 \
  --confusion-type subject_confusion
```

## 可选 LLM 生成与审计

默认生成路径是确定性的。若想让模型生成更自然的 query，或在生成时验证 QA 和
hard negative，可以启用 LLM。RAGProbe 支持 Qwen 预设，也支持通用
OpenAI-compatible chat completions API。

环境变量默认读取 `AI_API_KEY`；如果你想使用 `DASHSCOPE_API_KEY`、
`OPENAI_API_KEY` 或团队内部统一的环境变量名，可以通过 `--api-key-env` 指定：

```bash
export AI_API_KEY="..."
```

Windows PowerShell：

```powershell
$env:AI_API_KEY="..."
```

Qwen 示例：

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/qwen-testset.json \
  --llm qwen \
  --model qwen-plus \
  --llm-validate \
  --yes \
  --quality-report .tmp/qwen-quality.md
```

OpenAI-compatible 示例：

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/llm-testset.json \
  --llm openai-compatible \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  --model qwen-plus \
  --api-key-env AI_API_KEY \
  --yes
```

例如使用 DashScope 常见的 `DASHSCOPE_API_KEY`：

```bash
python -m ragprobe generate \
  --chunks examples/contract/chunks.jsonl \
  --output .tmp/qwen-testset.json \
  --llm qwen \
  --model qwen-plus \
  --api-key-env DASHSCOPE_API_KEY \
  --yes
```

指定领域上下文（`--domain-hint`）可以让 LLM 生成更贴合领域的 query 风格：

```bash
python -m ragprobe generate \
  --chunks examples/medical/chunks.jsonl \
  --output .tmp/medical-testset.json \
  --llm qwen \
  --model qwen-plus \
  --domain-hint "医疗器械注册审评文档" \
  --yes
```

不传 `--domain-hint` 时，LLM 会从 chunk 内容自动推断语言和风格。

注意：

- 不要把 API key 写进代码、测试集、缓存或提交记录。
- `.ragprobe_cache/` 默认用于缓存 LLM 调用结果，已经建议加入 `.gitignore`。
- `diagnose`、`compare`、`check` 仍然不需要 LLM。

测试集审计：

```bash
python -m ragprobe audit \
  --testset examples/contract/testset.json \
  --output .tmp/audit.json \
  --markdown .tmp/audit.md \
  --llm qwen \
  --model qwen-plus \
  --sample-size 5
```

生成可人工审核的修复计划：

```bash
python -m ragprobe repair-plan \
  --audit-report .tmp/audit.json \
  --output .tmp/repair-plan.json \
  --markdown .tmp/repair-plan.md
```

应用安全修复到新测试集文件：

```bash
python -m ragprobe apply-audit-fixes \
  --testset examples/contract/testset.json \
  --repair-plan .tmp/repair-plan.json \
  --output .tmp/fixed-testset.json \
  --report .tmp/repair-apply.md
```

## 跨语言 retriever 接入

RAGProbe 不要求你的 RAG 系统用 Python 写。只要能把 query 转成 retrieved chunks，
就可以接入。

### 方式一：Python 文件

提供一个暴露 `retrieve(query, top_k)` 的 Python 文件：

```python
def retrieve(query: str, top_k: int = 10) -> list[dict]:
    return [
        {
            "chunk_id": "buyer_payment_30",
            "content": "买方逾期付款超过30天，应按未付款金额支付违约金。",
            "score": 0.95,
            "metadata": {"source": "contract.md"},
        }
    ][:top_k]
```

运行：

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --retriever examples/contract/python_retriever.py \
  --output .tmp/python-results.json
```

### 方式二：Node.js 或任意 JSONL 子进程

RAGProbe 会向子进程 stdin 逐行写入请求：

```jsonl
{"query":"买方逾期付款超过30天的违约金是多少？","top_k":10}
```

子进程需要向 stdout 逐行返回 JSON 数组：

```jsonl
[{"chunk_id":"buyer_payment_30","content":"买方逾期付款超过30天，应按未付款金额支付违约金。","score":0.95}]
```

这个协议对任何语言都一样：Java、Go、Rust、C#、PHP 只要能读 stdin、写 stdout
JSONL，就能接入。

Node.js 最小示例：

```javascript
const readline = require("readline");

const chunks = [
  {
    chunk_id: "buyer_payment_30",
    content: "买方逾期付款超过30天，应按未付款金额支付违约金。",
  },
  {
    chunk_id: "seller_delivery_15",
    content: "卖方延期交货超过15日，应承担延期交货违约责任。",
  },
];

const rl = readline.createInterface({ input: process.stdin });

rl.on("line", (line) => {
  const request = JSON.parse(line);
  const topK = request.top_k || 10;
  const results = chunks
    .map((chunk) => ({
      ...chunk,
      score: request.query.includes("买方") && chunk.chunk_id.includes("buyer") ? 1.0 : 0.3,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
  console.log(JSON.stringify(results));
});
```

Java 最小示例：

```java
import java.io.*;
import org.json.*;

public class RagProbeRetriever {
    public static void main(String[] args) throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = reader.readLine()) != null) {
            JSONObject req = new JSONObject(line);
            String query = req.getString("query");
            int topK = req.getInt("top_k");

            // 替换为你的实际检索逻辑
            JSONArray results = search(query, topK);
            System.out.println(results.toString());
        }
    }

    static JSONArray search(String query, int topK) {
        // 调用 Milvus / ES / 你的检索服务
        JSONArray arr = new JSONArray();
        arr.put(new JSONObject()
            .put("chunk_id", "algo_001")
            .put("content", "二分查找是一种在有序数组中查找元素的算法...")
            .put("score", 0.95));
        return arr;
    }
}
```

运行：

```bash
# Node.js
python -m ragprobe run \
  --testset testset.json \
  --retriever-cmd "node my_retriever.js" \
  --output .tmp/node-results.json

# Java
python -m ragprobe run \
  --testset testset.json \
  --retriever-cmd "java -cp your-app.jar RagProbeRetriever" \
  --output .tmp/java-results.json
```

### 方式三：HTTP endpoint

适合直接测试已运行的 Java/Go/Node.js 服务，不需要写任何适配代码。

单条请求协议：

```http
POST /search
Content-Type: application/json

{"query":"买方逾期付款超过30天的违约金是多少？","top_k":10}
```

返回：

```json
[
  {
    "chunk_id": "buyer_payment_30",
    "content": "买方逾期付款超过30天，应按未付款金额支付违约金。",
    "score": 0.95
  }
]
```

CLI 运行（不需要配置文件）：

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --endpoint http://127.0.0.1:8008/search \
  --output .tmp/http-results.json
```

Python API 运行（所有参数直接传入）：

```python
from ragprobe import RAGProbe

probe = RAGProbe()
result = probe.pipeline(
    testset="testset.json",
    endpoint="http://localhost:8080/search",
    headers={"Authorization": "Bearer dev-token"},
    timeout=10,
    top_k=5,
    min_hit_rate=0.7,
)
```

如果你的服务返回格式不同（比如 Spring Boot 常见的嵌套结构），只需加一个薄
Controller 做格式转换：

```java
// Spring Boot 适配示例
@PostMapping("/ragprobe/search")
public List<Map<String, Object>> search(@RequestBody Map<String, Object> req) {
    String query = (String) req.get("query");
    int topK = (int) req.getOrDefault("top_k", 10);

    List<SearchResult> results = ragService.search(query, topK);

    return results.stream().map(r -> Map.of(
        "chunk_id", r.getId(),
        "content", r.getContent(),
        "score", r.getScore()
    )).collect(Collectors.toList());
}
```

仍然支持 `endpoint_config.json`（向后兼容）：

```json
{
  "headers": {
    "Authorization": "Bearer dev-token"
  },
  "timeout": 30,
  "batch_size": 2
}
```

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --endpoint http://127.0.0.1:8008/search \
  --endpoint-config examples/contract/endpoint_config.json \
  --output .tmp/http-results.json
```

## Python API

### 一站式 Pipeline

`pipeline()` 方法将数据准备、检索执行、诊断分析、CI 检查合并为一次调用，
所有配置通过参数传入，不需要任何 JSON 配置文件：

```python
from ragprobe import RAGProbe

probe = RAGProbe()

# 最简用法：已有 testset + 内置 baseline
result = probe.pipeline(
    testset="testset.json",
    baseline="lexical",
)
print(result.report.hit_rate, result.report.mrr)

# 测 Java/Spring Boot 服务，无需配置文件
result = probe.pipeline(
    testset="algorithm_testset.json",
    endpoint="http://localhost:8080/search",
    headers={"Authorization": "Bearer your-token"},
    top_k=5,
    min_hit_rate=0.85,
    max_fpr=0.10,
)
if not result.check.passed:
    print("检索质量未达标！")

# 从 chunks 自动生成 testset + 跑检索 + 出报告 + 保存所有产物
result = probe.pipeline(
    chunks="algorithm_chunks.jsonl",
    endpoint="http://localhost:8080/search",
    headers={"X-Api-Key": "dev-key"},
    num_cases=20,
    top_k=10,
    output_dir="./ragprobe_output",
)

# 未来接数据库 connector（source 对象需暴露 export() 方法）
result = probe.pipeline(
    source=your_milvus_source,
    endpoint="http://localhost:8080/search",
    min_hit_rate=0.80,
)
```

`pipeline()` 返回 `PipelineResult`，包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `testset` | `TestSet` | 使用的测试集（传入的或自动生成的） |
| `results` | `list[RetrievalResult]` | 每条 query 的检索结果 |
| `report` | `DiagnosticReport` | 诊断报告（hit_rate, mrr, fpr 等） |
| `check` | `CheckResult \| None` | CI 阈值检查结果（传了阈值参数才有） |

传 `output_dir` 时自动保存 `testset.json`、`results.json`、`report.json`、`report.md`。

### 分步调用

如果需要更细粒度的控制，可以分步调用：

```python
from ragprobe import RAGProbe

probe = RAGProbe()

testset = probe.generate(
    chunks="examples/contract/chunks.jsonl",
    hard_negative_top_k=2,
)

# LLM 生成时可指定领域上下文
testset = probe.generate(
    chunks="examples/medical/chunks.jsonl",
    llm="qwen",
    domain_hint="医疗器械注册审评文档",
)

# AI 参数既可以在初始化时作为默认值传入
probe = RAGProbe(
    llm="qwen",
    model="qwen-plus",
    api_key_env="DASHSCOPE_API_KEY",
)

testset = probe.generate(
    chunks="examples/contract/chunks.jsonl",
    llm_validate=True,
)

# 本地快速试用时，也可以直接传 api_key；不要把真实 key 提交到仓库
probe = RAGProbe(
    llm="qwen",
    model="qwen-plus",
    api_key="sk-...",
)

# 也可以在单次调用时覆盖 AI 参数
testset = probe.generate(
    chunks="examples/contract/chunks.jsonl",
    llm="openai-compatible",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    model="qwen-plus",
    api_key="sk-...",
    api_key_env="DASHSCOPE_API_KEY",
    llm_validate=True,
)

audit = probe.audit(
    testset=testset,
    llm="qwen",
    model="qwen-plus",
    api_key_env="DASHSCOPE_API_KEY",
    sample_size=5,
)

# run() 现在支持直接传 headers，不再需要 endpoint_config.json
results = probe.run(
    testset=testset,
    endpoint="http://localhost:8080/search",
    headers={"Authorization": "Bearer token"},
    timeout=10,
    top_k=5,
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

内置 baseline：

```python
results = probe.run(
    testset="examples/contract/testset.json",
    baseline="embedding",
    top_k=10,
)
```

多 retriever 实验：

```python
report = probe.experiment(
    config="examples/contract/experiment.json",
    output_dir=".tmp/contract-experiment",
)
```

## 数据源接入模板

`pipeline(source=xxx)` 接受任何暴露 `export()` 方法的对象，返回 chunk 列表。
以下是各种常见 RAG 数据源的接入模板，复制后修改字段名即可使用。

### source 协议

```python
class YourSource:
    def export(self) -> list[dict]:
        """返回 chunk 列表，每个 chunk 至少包含 chunk_id 和 content。"""
        return [
            {
                "chunk_id": "unique_id",
                "content": "chunk 文本内容",
                "metadata": {"key": "value"},       # 可选
                "source_document": "来源文件名",     # 可选
            }
        ]
```

使用方式统一为：

```python
from ragprobe import RAGProbe

probe = RAGProbe()
result = probe.pipeline(
    source=YourSource(...),
    endpoint="http://localhost:8080/search",
    min_hit_rate=0.85,
)
```

---

### Milvus

```python
class MilvusSource:
    def __init__(self, uri, collection, content_field="content",
                 id_field="chunk_id", metadata_fields=None, limit=10000):
        self.uri = uri
        self.collection = collection
        self.content_field = content_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields or []
        self.limit = limit

    def export(self):
        from pymilvus import connections, Collection

        connections.connect(uri=self.uri)
        col = Collection(self.collection)
        col.load()

        output_fields = [self.id_field, self.content_field] + self.metadata_fields
        results = col.query(expr="", output_fields=output_fields, limit=self.limit)

        chunks = []
        for r in results:
            chunks.append({
                "chunk_id": str(r[self.id_field]),
                "content": r[self.content_field],
                "metadata": {f: r.get(f) for f in self.metadata_fields if r.get(f) is not None},
            })
        connections.disconnect("default")
        return chunks
```

使用：

```python
source = MilvusSource(
    uri="http://localhost:19530",
    collection="algorithm_knowledge",
    content_field="text",
    id_field="doc_id",
    metadata_fields=["topic", "category", "difficulty"],
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### Elasticsearch / OpenSearch

```python
class ElasticsearchSource:
    def __init__(self, hosts, index, content_field="content",
                 id_field="_id", metadata_fields=None, query=None, size=10000):
        self.hosts = hosts
        self.index = index
        self.content_field = content_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields or []
        self.query = query or {"match_all": {}}
        self.size = size

    def export(self):
        from elasticsearch import Elasticsearch, helpers

        es = Elasticsearch(self.hosts)
        fields = [self.content_field] + self.metadata_fields

        chunks = []
        for hit in helpers.scan(es, index=self.index, query={"query": self.query},
                                _source=fields, size=self.size):
            source = hit["_source"]
            chunk_id = hit[self.id_field] if self.id_field == "_id" else source.get(self.id_field, hit["_id"])
            chunks.append({
                "chunk_id": str(chunk_id),
                "content": source[self.content_field],
                "metadata": {f: source[f] for f in self.metadata_fields if f in source},
            })
        return chunks
```

使用：

```python
source = ElasticsearchSource(
    hosts=["http://localhost:9200"],
    index="algorithm_docs",
    content_field="text",
    metadata_fields=["topic", "source_file"],
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### Qdrant

```python
class QdrantSource:
    def __init__(self, url, collection, content_key="content",
                 id_key=None, limit=10000):
        self.url = url
        self.collection = collection
        self.content_key = content_key
        self.id_key = id_key
        self.limit = limit

    def export(self):
        from qdrant_client import QdrantClient

        client = QdrantClient(url=self.url)
        points, offset = [], None

        while True:
            result = client.scroll(
                collection_name=self.collection,
                limit=min(self.limit, 1000),
                offset=offset,
                with_payload=True,
            )
            batch, offset = result
            if not batch:
                break
            points.extend(batch)
            if offset is None or len(points) >= self.limit:
                break

        chunks = []
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get(self.id_key, str(point.id)) if self.id_key else str(point.id)
            content = payload.get(self.content_key, "")
            metadata = {k: v for k, v in payload.items() if k not in (self.content_key, self.id_key)}
            chunks.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})
        return chunks
```

使用：

```python
source = QdrantSource(
    url="http://localhost:6333",
    collection="algorithm_knowledge",
    content_key="text",
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### Redis（向量检索模式）

```python
class RedisSource:
    def __init__(self, url, prefix="doc:", content_field="content",
                 id_field=None, metadata_fields=None, limit=10000):
        self.url = url
        self.prefix = prefix
        self.content_field = content_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields or []
        self.limit = limit

    def export(self):
        import redis

        r = redis.from_url(self.url)
        cursor, chunks = 0, []

        while True:
            cursor, keys = r.scan(cursor=cursor, match=f"{self.prefix}*", count=500)
            for key in keys:
                if len(chunks) >= self.limit:
                    break
                data = r.hgetall(key)
                if not data:
                    continue
                decoded = {k.decode(): v.decode() for k, v in data.items()
                           if k.decode() != "embedding"}
                chunk_id = decoded.get(self.id_field, key.decode().removeprefix(self.prefix)) if self.id_field else key.decode().removeprefix(self.prefix)
                content = decoded.get(self.content_field, "")
                metadata = {f: decoded[f] for f in self.metadata_fields if f in decoded}
                chunks.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})
            if cursor == 0 or len(chunks) >= self.limit:
                break
        return chunks
```

使用：

```python
source = RedisSource(
    url="redis://localhost:6379",
    prefix="algo:",
    content_field="text",
    metadata_fields=["topic", "difficulty"],
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### PostgreSQL（pgvector 或纯文本检索）

```python
class PostgresSource:
    def __init__(self, dsn, table, content_column="content",
                 id_column="id", metadata_columns=None, where=None, limit=10000):
        self.dsn = dsn
        self.table = table
        self.content_column = content_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns or []
        self.where = where
        self.limit = limit

    def export(self):
        import psycopg2

        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()

        columns = [self.id_column, self.content_column] + self.metadata_columns
        sql = f"SELECT {', '.join(columns)} FROM {self.table}"
        if self.where:
            sql += f" WHERE {self.where}"
        sql += f" LIMIT {self.limit}"

        cur.execute(sql)
        col_names = [desc[0] for desc in cur.description]

        chunks = []
        for row in cur.fetchall():
            record = dict(zip(col_names, row))
            chunks.append({
                "chunk_id": str(record[self.id_column]),
                "content": record[self.content_column],
                "metadata": {c: record[c] for c in self.metadata_columns if record.get(c) is not None},
            })
        cur.close()
        conn.close()
        return chunks
```

使用：

```python
source = PostgresSource(
    dsn="postgresql://user:pass@localhost:5432/ragdb",
    table="knowledge_chunks",
    content_column="chunk_text",
    id_column="chunk_id",
    metadata_columns=["topic", "source_file", "created_at"],
    where="category = 'algorithm'",
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### MySQL / MariaDB

```python
class MySQLSource:
    def __init__(self, host, database, table, content_column="content",
                 id_column="id", metadata_columns=None, where=None,
                 user="root", password="", port=3306, limit=10000):
        self.host = host
        self.database = database
        self.table = table
        self.content_column = content_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns or []
        self.where = where
        self.user = user
        self.password = password
        self.port = port
        self.limit = limit

    def export(self):
        import pymysql

        conn = pymysql.connect(
            host=self.host, port=self.port, user=self.user,
            password=self.password, database=self.database, charset="utf8mb4",
        )
        cur = conn.cursor(pymysql.cursors.DictCursor)

        columns = [self.id_column, self.content_column] + self.metadata_columns
        sql = f"SELECT {', '.join(columns)} FROM {self.table}"
        if self.where:
            sql += f" WHERE {self.where}"
        sql += f" LIMIT {self.limit}"

        cur.execute(sql)
        chunks = []
        for row in cur.fetchall():
            chunks.append({
                "chunk_id": str(row[self.id_column]),
                "content": row[self.content_column],
                "metadata": {c: row[c] for c in self.metadata_columns if row.get(c) is not None},
            })
        cur.close()
        conn.close()
        return chunks
```

使用：

```python
source = MySQLSource(
    host="localhost",
    database="rag_system",
    table="algorithm_chunks",
    content_column="text",
    id_column="chunk_id",
    metadata_columns=["topic", "difficulty"],
    where="collection_name = 'algorithm'",
    user="root",
    password="your_password",
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### MongoDB

```python
class MongoSource:
    def __init__(self, uri, database, collection, content_field="content",
                 id_field="_id", metadata_fields=None, filter=None, limit=10000):
        self.uri = uri
        self.database = database
        self.collection = collection
        self.content_field = content_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields or []
        self.filter = filter or {}
        self.limit = limit

    def export(self):
        from pymongo import MongoClient

        client = MongoClient(self.uri)
        col = client[self.database][self.collection]

        projection = {self.content_field: 1, self.id_field: 1}
        for f in self.metadata_fields:
            projection[f] = 1

        chunks = []
        for doc in col.find(self.filter, projection).limit(self.limit):
            chunk_id = str(doc.get(self.id_field, doc.get("_id")))
            chunks.append({
                "chunk_id": chunk_id,
                "content": doc.get(self.content_field, ""),
                "metadata": {f: doc[f] for f in self.metadata_fields if f in doc},
            })
        client.close()
        return chunks
```

使用：

```python
source = MongoSource(
    uri="mongodb://localhost:27017",
    database="rag",
    collection="algorithm_knowledge",
    content_field="text",
    metadata_fields=["topic", "tags"],
    filter={"status": "active"},
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### Weaviate

```python
class WeaviateSource:
    def __init__(self, url, class_name, content_property="content",
                 metadata_properties=None, limit=10000):
        self.url = url
        self.class_name = class_name
        self.content_property = content_property
        self.metadata_properties = metadata_properties or []
        self.limit = limit

    def export(self):
        import weaviate

        client = weaviate.Client(self.url)
        properties = [self.content_property] + self.metadata_properties

        result = (
            client.query
            .get(self.class_name, properties)
            .with_additional(["id"])
            .with_limit(self.limit)
            .do()
        )

        objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])
        chunks = []
        for obj in objects:
            chunk_id = obj.get("_additional", {}).get("id", "")
            chunks.append({
                "chunk_id": chunk_id,
                "content": obj.get(self.content_property, ""),
                "metadata": {p: obj[p] for p in self.metadata_properties if p in obj},
            })
        return chunks
```

使用：

```python
source = WeaviateSource(
    url="http://localhost:8080",
    class_name="AlgorithmChunk",
    content_property="text",
    metadata_properties=["topic", "difficulty"],
)
result = probe.pipeline(source=source, endpoint="http://localhost:9090/search")
```

---

### 本地文件目录（非向量化，适合 lexical / grep 类检索）

```python
class FileDirectorySource:
    def __init__(self, directory, glob_pattern="**/*.md",
                 chunk_by="file", max_chunk_chars=1000):
        self.directory = directory
        self.glob_pattern = glob_pattern
        self.chunk_by = chunk_by
        self.max_chunk_chars = max_chunk_chars

    def export(self):
        from pathlib import Path

        base = Path(self.directory)
        chunks = []

        for filepath in sorted(base.glob(self.glob_pattern)):
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            rel_path = str(filepath.relative_to(base))

            if self.chunk_by == "file":
                chunks.append({
                    "chunk_id": rel_path,
                    "content": text[:self.max_chunk_chars],
                    "metadata": {"source_file": rel_path},
                    "source_document": rel_path,
                })
            elif self.chunk_by == "paragraph":
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                for i, para in enumerate(paragraphs):
                    chunks.append({
                        "chunk_id": f"{rel_path}#p{i}",
                        "content": para[:self.max_chunk_chars],
                        "metadata": {"source_file": rel_path, "paragraph": i},
                        "source_document": rel_path,
                    })
        return chunks
```

使用：

```python
# 适合测试基于文件的 grep / lexical 检索系统
source = FileDirectorySource(
    directory="./docs/algorithms",
    glob_pattern="**/*.md",
    chunk_by="paragraph",
    max_chunk_chars=2000,
)
result = probe.pipeline(source=source, baseline="lexical")
```

---

### HTTP API 导出（从已有管理接口拉取）

```python
class HTTPExportSource:
    def __init__(self, url, headers=None, id_path="id",
                 content_path="content", metadata_paths=None):
        self.url = url
        self.headers = headers or {}
        self.id_path = id_path
        self.content_path = content_path
        self.metadata_paths = metadata_paths or []

    def export(self):
        import urllib.request
        import json

        req = urllib.request.Request(self.url, headers=self.headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items = data if isinstance(data, list) else data.get("items", data.get("data", []))
        chunks = []
        for item in items:
            chunks.append({
                "chunk_id": str(self._get_nested(item, self.id_path)),
                "content": str(self._get_nested(item, self.content_path)),
                "metadata": {p: self._get_nested(item, p) for p in self.metadata_paths
                             if self._get_nested(item, p) is not None},
            })
        return chunks

    @staticmethod
    def _get_nested(obj, path):
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return None
        return obj
```

使用：

```python
# 从你的 Java 管理端接口拉取知识库
source = HTTPExportSource(
    url="http://localhost:8080/internal/knowledge/chunks?collection=algorithm",
    headers={"Authorization": "Bearer admin-token"},
    id_path="chunkId",
    content_path="content",
    metadata_paths=["topic", "metadata.category"],
)
result = probe.pipeline(source=source, endpoint="http://localhost:8080/search")
```

---

### 自定义 source

任何满足 `export()` 协议的对象都可以作为 source：

```python
class CustomSource:
    def export(self):
        # 你的任意逻辑：调 gRPC、读 Parquet、查 SQLite...
        return [
            {"chunk_id": "1", "content": "...", "metadata": {}},
            {"chunk_id": "2", "content": "...", "metadata": {}},
        ]

result = probe.pipeline(source=CustomSource(), baseline="lexical")
```

## 对比、实验与 CI

```bash
python -m ragprobe compare \
  --testset examples/contract/testset.json \
  --before .tmp/old-results.json \
  --after .tmp/new-results.json
```

多 retriever 实验：

```bash
python -m ragprobe experiment \
  --config examples/contract/experiment.json \
  --output-dir .tmp/contract-experiment
```

CI 阈值检查：

```bash
python -m ragprobe check \
  --testset examples/contract/testset.json \
  --results .tmp/contract-results.json \
  --min-hit-rate 0.7 \
  --min-mrr 0.5 \
  --max-fpr 0.3
```

## 内置 baseline

```bash
python -m ragprobe run \
  --testset examples/contract/testset.json \
  --baseline lexical \
  --output .tmp/lexical-baseline-results.json

python -m ragprobe run \
  --testset examples/contract/testset.json \
  --baseline embedding \
  --output .tmp/embedding-baseline-results.json
```

- `lexical`：token overlap scoring。
- `embedding`：本地 hashed token-vector cosine scoring。

这两个 baseline 不下载模型、不调用 API，适合作为 CI 和实验中的稳定对照组。


## 与其他工具的关系

RAGAS、DeepEval、TruLens 评估的是 RAG 管线的最终输出（答案质量、忠实度、相关性）。
RAGProbe 评估的是更靠前的检索层：retriever 是否找对了 chunk、是否抗住了 hard negative。
二者互补，不冲突。

## RAGProbe 不做什么

- 不评估最终 LLM answer 的文本质量。
- 不是 RAG framework。
- 不是 vector database。
- 不是实时监控 dashboard。
- 核心诊断闭环不依赖 LLM judge。



## License

MIT
