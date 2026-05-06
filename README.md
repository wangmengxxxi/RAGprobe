# RAGProbe

> Diagnose your RAG before your users do.

## 问题

你的合同 RAG 系统，用户问：

```
"买方逾期付款超过30天的违约金是多少？"
```

系统返回了：

```
"卖方延期交货超过15日，应按合同总额的0.03%/日支付违约金。"
```

主体错了（卖方 ≠ 买方），事件错了（交货 ≠ 付款），条件也错了（15日 ≠ 30天）。但它们的 embedding 相似度是 0.94。

用户没仔细看，直接用了。

**RAGProbe 能在上线前帮你发现这类问题。**

## 它做什么

```
你的文档 → 自动生成测试集（含 hard negative）
         → 跑你的 retrieval 系统
         → 告诉你哪类问题会出错、为什么、怎么改
```

不只给一个分数。给你一份诊断报告：

```
⚠️  Hard Negative Resistance
  FPR: 0.35 (35% of hard negatives incorrectly retrieved)
  
  Confusion Type Distribution:
    subject_confusion:    40% ████████░░
    condition_confusion:  25% █████░░░░░
    event_confusion:      20% ████░░░░░░

💡 Recommendations
  1. Add cross-encoder reranker (expected FPR reduction: 50%+)
  2. Reduce chunk size (20% of misses due to oversized chunks)
```

## Quick Start

```bash
pip install ragprobe

# 1. 从你的文档生成测试集
ragprobe generate --docs ./contracts/ --output testset.json

# 2. 跑你的 retriever
ragprobe run --testset testset.json --retriever my_retriever.py --output results.json

# 3. 看诊断报告
ragprobe diagnose --results results.json
```

或者用 Python：

```python
from ragprobe import RAGProbe

probe = RAGProbe(llm="gpt-4o-mini", embedding="bge-large-zh")

# 生成测试集
testset = probe.generate_testset(documents=["contract.pdf"])

# 跑你的 retriever
results = probe.run(testset=testset, retriever=my_retriever)

# 诊断
report = probe.diagnose(results)
report.print()
```

## Retriever 接口

你只需要提供一个函数：

```python
def my_retriever(query: str, top_k: int = 10) -> list[dict]:
    """
    返回格式：[{"content": "...", "score": 0.85, "metadata": {...}}, ...]
    """
    # 你的检索逻辑
    results = your_vector_db.search(query, top_k=top_k)
    return results
```

也支持离线评估（直接提供 JSON 格式的检索结果）。

## 与 RAGAS 的区别

| | RAGAS | RAGProbe |
|---|---|---|
| 核心能力 | 通用 RAG / LLM 应用评估 | 检索失败归因 + 诊断建议 |
| 关注重点 | faithfulness、relevancy、context metrics | hard negative、混淆类型、检索失败模式 |
| 测试集 | 支持 synthetic testset | 专门生成/挖掘 hard negative testset |
| 输出 | 指标分数 | worst cases、confusion distribution、改进建议 |
| LLM 使用 | 常用于评分 | 尽量让 retrieval 指标可复现，LLM 主要用于生成/归因 |
| 中文场景 | 通用支持 | 优先提供中文 hard negative 模板和示例 |

## 它不做什么

- 不评估 LLM 生成质量（只管 retrieval）
- 不是 RAG 框架（不提供检索能力）
- 不是监控平台（不做实时告警）

## 适用场景

你的文档里存在大量 "看起来相似但不该被混淆" 的内容：

- 合同条款（买方 vs 卖方、不同违约条件）
- 医疗指南（成人 vs 儿童剂量、禁忌 vs 适应症）
- 金融产品（不同产品的相似条款）
- 技术手册（不同型号的相似参数）

## License

MIT
