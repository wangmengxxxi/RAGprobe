def retrieve(query: str, top_k: int = 10) -> list[dict]:
    if "买方" in query:
        return [
            {
                "chunk_id": "buyer_payment_30",
                "content": "买方逾期付款条款",
                "score": 0.95,
            }
        ][:top_k]
    return [
        {
            "chunk_id": "seller_delivery_15",
            "content": "卖方延期交货条款",
            "score": 0.96,
        }
    ][:top_k]
