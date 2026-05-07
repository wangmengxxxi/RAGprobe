def retrieve(query: str, top_k: int = 10) -> list[dict]:
    if "买方" in query:
        return [
            {
                "content": "买方逾期付款条款",
                "score": 0.95,
            }
        ][:top_k]
    return [
        {
            "content": "卖方延期交货条款",
            "score": 0.96,
        }
    ][:top_k]
