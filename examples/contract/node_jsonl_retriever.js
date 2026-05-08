const readline = require("readline");

const chunks = {
  buyer_payment_30: "买方逾期付款超过30天，应按未付款金额支付违约金。",
  seller_delivery_15: "卖方延期交货超过15日，应承担延期交货违约责任。",
  buyer_payment_notice: "买方收到付款通知后，应在10日内完成付款确认。",
  seller_invoice_notice: "卖方开具发票后，应在3日内通知买方付款。"
};

function retrieve(query, topK) {
  let ids;
  if (query.includes("逾期付款") || query.includes("违约金")) {
    ids = ["buyer_payment_30"];
  } else if (query.includes("延期交货")) {
    ids = ["seller_delivery_15"];
  } else if (query.includes("付款通知") || query.includes("确认付款")) {
    ids = ["buyer_payment_notice"];
  } else {
    ids = ["seller_invoice_notice"];
  }

  return ids.slice(0, topK).map((chunkId, index) => ({
    chunk_id: chunkId,
    content: chunks[chunkId],
    score: 1.0 - index * 0.1,
    metadata: { example: "contract" }
  }));
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on("line", (line) => {
  const payload = JSON.parse(line);
  const results = retrieve(payload.query, payload.top_k || 10);
  process.stdout.write(`${JSON.stringify(results)}\n`);
});
