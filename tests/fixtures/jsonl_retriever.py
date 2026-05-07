from __future__ import annotations

import json
import sys


for line in sys.stdin:
    payload = json.loads(line)
    query = payload["query"]
    if "买方" in query:
        results = [{"chunk_id": "buyer_payment_30", "content": "买方逾期付款条款", "score": 0.95}]
    else:
        results = [{"chunk_id": "seller_delivery_15", "content": "卖方延期交货条款", "score": 0.96}]
    print(json.dumps(results, ensure_ascii=False), flush=True)
