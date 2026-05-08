from __future__ import annotations


CHUNKS = {
    "buyer_payment_30": "买方逾期付款超过30天，应按未付款金额支付违约金。",
    "seller_delivery_15": "卖方延期交货超过15日，应承担延期交货违约责任。",
    "buyer_payment_notice": "买方收到付款通知后，应在10日内完成付款确认。",
    "seller_invoice_notice": "卖方开具发票后，应在3日内通知买方付款。",
}


def retrieve(query: str, top_k: int = 10) -> list[dict]:
    """Intentionally weak retriever for compare demos."""
    if "逾期付款" in query or "违约金" in query:
        ids = ["seller_delivery_15", "buyer_payment_30"]
    elif "延期交货" in query:
        ids = ["buyer_payment_30", "seller_delivery_15"]
    elif "付款通知" in query or "确认付款" in query:
        ids = ["seller_invoice_notice", "buyer_payment_notice"]
    else:
        ids = ["buyer_payment_notice", "seller_invoice_notice"]

    return [
        {
            "chunk_id": chunk_id,
            "content": CHUNKS[chunk_id],
            "score": 1.0 - index * 0.1,
            "metadata": {"example": "contract", "quality": "weak"},
        }
        for index, chunk_id in enumerate(ids[:top_k])
    ]
