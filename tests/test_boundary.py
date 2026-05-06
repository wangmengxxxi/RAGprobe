from __future__ import annotations

import unittest
from pathlib import Path

from qualira.backends.sqlite import SQLiteEvidenceBackend, load_evidence_file
from qualira.eval.benchmark import run_benchmark
from qualira.retrieval.executor import RetrievalExecutor


FIXTURES = Path(__file__).parent / "fixtures"


class BoundaryRetrievalTest(unittest.TestCase):
    def test_boundary_excludes_subject_event_hard_negative(self) -> None:
        backend = SQLiteEvidenceBackend(":memory:")
        backend.store_many(load_evidence_file(FIXTURES / "contract_evidence.json"))

        result = RetrievalExecutor(backend).execute("买方逾期付款超过30天有什么责任？", limit=5)

        selected = [item.unit.id for item in result.selected]
        excluded = [item.unit_id for item in result.excluded]
        self.assertIn("ev_001_buyer_late_payment_30_liability", selected)
        self.assertIn("ev_002_seller_late_delivery_15_liability", excluded)
        self.assertNotIn("ev_002_seller_late_delivery_15_liability", selected)

    def test_phase0_benchmark_beats_naive_fpr_without_losing_ehr(self) -> None:
        metrics, _details, _table = run_benchmark(
            FIXTURES / "contract_evidence.json",
            FIXTURES / "contract_queries.json",
            top_k=5,
        )
        by_method = {metric.method: metric for metric in metrics}
        naive = by_method["naive_chunk_topk"]
        qualira = by_method["qualira"]

        self.assertLess(qualira.false_positive_rate, naive.false_positive_rate)
        self.assertGreaterEqual(qualira.evidence_hit_rate, naive.evidence_hit_rate - 0.05)
        self.assertIsNotNone(qualira.boundary_exclusion_rate)
        self.assertGreater(qualira.boundary_exclusion_rate or 0.0, 0.5)
        self.assertEqual(qualira.source_grounding_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
