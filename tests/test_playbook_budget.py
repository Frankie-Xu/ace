import json
import tempfile
import unittest
from unittest.mock import patch

from ace.ace import ACE
from ace.ace_batch import ACEBatch
from ace.core.curator import Curator
from playbook_utils import (
    apply_curator_operations,
    apply_curator_operations_with_budget,
)
from utils import count_tokens


class PlaybookBudgetTest(unittest.TestCase):
    def setUp(self):
        self.playbook = "## STRATEGIES & INSIGHTS\n"
        self.operation = {
            "type": "ADD",
            "section": "STRATEGIES & INSIGHTS",
            "content": "A concise synthetic budget test bullet.",
        }
        self.operations = [self.operation]
        self.candidate, self.next_id = apply_curator_operations(
            self.playbook, self.operations, 7
        )

    def test_exact_boundary_is_accepted_without_changing_id_policy(self):
        budget = count_tokens(self.candidate)
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, self.operations, 7, budget, enforce=True
        )
        self.assertEqual(result, self.candidate)
        self.assertEqual(next_id, self.next_id)
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["base_tokens"], count_tokens(self.playbook))
        self.assertEqual(decision["candidate_tokens"], budget)
        self.assertEqual(decision["budget_tokens"], budget)
        self.assertEqual(decision["reason"], "within_budget")

    def test_over_budget_returns_original_playbook_and_does_not_consume_id(self):
        budget = count_tokens(self.candidate) - 1
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, self.operations, 7, budget, enforce=True
        )
        self.assertEqual(result, self.playbook)
        self.assertEqual(next_id, 7)
        self.assertEqual(decision["status"], "rejected")
        self.assertEqual(decision["reason"], "candidate_exceeds_budget")
        self.assertGreater(decision["candidate_tokens"], decision["budget_tokens"])

    def test_empty_operations_are_accepted_and_do_not_allocate_id(self):
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, [], 7, 0, enforce=True
        )
        self.assertEqual(result, self.playbook)
        self.assertEqual(next_id, 7)
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["reason"], "no_operations")

    def test_disabled_enforcement_preserves_existing_behavior(self):
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, self.operations, 7, 0, enforce=False
        )
        self.assertEqual(result, self.candidate)
        self.assertEqual(next_id, self.next_id)
        self.assertEqual(decision["enforced"], False)
        self.assertEqual(decision["reason"], "enforcement_disabled")

    def test_curator_rejection_is_structured_and_state_unchanged(self):
        with tempfile.TemporaryDirectory() as log_dir:
            response = json.dumps({"reasoning": "synthetic", "operations": self.operations})
            budget = count_tokens(self.candidate) - 1
            with patch("ace.core.curator.timed_llm_call", return_value=(response, {})):
                result = Curator(None, "offline", "test").curate(
                    current_playbook=self.playbook,
                    recent_reflection="reflection",
                    question_context="context",
                    current_step=1,
                    total_samples=1,
                    token_budget=budget,
                    playbook_stats={},
                    call_id="budget-test",
                    log_dir=log_dir,
                    next_global_id=7,
                    enforce_playbook_budget=True,
                )
        self.assertEqual(result[0], self.playbook)
        self.assertEqual(result[1], 7)
        self.assertEqual(result[2], [])
        self.assertEqual(result[3]["playbook_budget"]["status"], "rejected")
        self.assertEqual(result[3]["playbook_budget"]["base_tokens"], count_tokens(self.playbook))

    def test_ace_and_batch_default_to_compatibility(self):
        self.assertFalse(ACE._extract_config_params(object.__new__(ACE), {})[
            "enforce_playbook_token_budget"
        ])
        self.assertFalse(ACEBatch._extract_config_params(object.__new__(ACEBatch), {})[
            "enforce_playbook_token_budget"
        ])
        self.assertTrue(ACE._extract_config_params(
            object.__new__(ACE), {"enforce_playbook_token_budget": True}
        )["enforce_playbook_token_budget"])
        self.assertTrue(ACEBatch._extract_config_params(
            object.__new__(ACEBatch), {"enforce_playbook_token_budget": True}
        )["enforce_playbook_token_budget"])

    def test_batch_reducer_over_budget_keeps_snapshot_and_next_id(self):
        budget = count_tokens(self.candidate) - 1
        result, next_id, decision = apply_curator_operations_with_budget(
            self.playbook, self.operations, 7, budget, enforce=True
        )
        self.assertEqual((result, next_id), (self.playbook, 7))
        self.assertEqual(decision["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
