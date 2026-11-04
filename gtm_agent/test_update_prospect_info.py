import importlib
import json
import os
import unittest
from unittest.mock import patch


with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "tech_stack": ["AWS"], "engagement_history": [],
            "account_details": [], "annual_revenue": 100000000,
        }
        records = patch.dict(agent.data_service.PROSPECTS, {
            "TEST-LEAD": self.record,
            "OTHER-LEAD": {**self.record, "tech_stack": ["AWS"]},
        }, clear=True)
        records.start()
        self.addCleanup(records.stop)
        profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        profiles.start()
        self.addCleanup(profiles.stop)

    def test_update_persists_stack_without_duplicates(self):
        result = agent.update_prospect_info.func("TEST-LEAD", "Kafka")
        self.assertEqual(result, {
            "prospect_id": "TEST-LEAD", "technology": "Kafka",
            "updated": True, "found": True, "tech_stack": ["AWS", "Kafka"],
        })
        self.assertEqual(self.record["tech_stack"], ["AWS", "Kafka"])
        self.assertEqual(agent.update_prospect_info.func("TEST-LEAD", "Kafka"), result)
        self.assertEqual(self.record["tech_stack"], ["AWS", "Kafka"])

    def test_unknown_prospect_does_not_report_success(self):
        self.assertEqual(agent.update_prospect_info.func("MISSING-LEAD", "Kafka"), {
            "prospect_id": "MISSING-LEAD", "technology": "Kafka",
            "updated": False, "found": False,
        })
        self.assertIsNone(agent.data_service.get_prospect_record("MISSING-LEAD"))

    def test_update_refreshes_only_changed_prospects_profile(self):
        original = agent.build_prospect_profile.func("TEST-LEAD")["prospect_profile"]
        other = agent.build_prospect_profile.func("OTHER-LEAD")["prospect_profile"]
        agent.update_prospect_info.func("TEST-LEAD", "Kafka")
        refreshed = agent.build_prospect_profile.func("TEST-LEAD")["prospect_profile"]
        self.assertEqual(refreshed["tech_stack"], ["AWS", "Kafka"])
        self.assertEqual(original["tech_stack"], ["AWS"])
        self.assertIs(agent.build_prospect_profile.func("OTHER-LEAD")["prospect_profile"], other)
        self.assertEqual(other["tech_stack"], ["AWS"])
        agent.update_prospect_info.func("TEST-LEAD", "Kafka")
        self.assertIs(agent.build_prospect_profile.func("TEST-LEAD")["prospect_profile"], refreshed)

    def test_update_profile_score_flow_uses_saved_stack(self):
        for technology in ("Kafka", "Terraform"):
            with self.subTest(technology=technology):
                offering = {
                    "required_tech_stack": ["AWS", technology],
                    "min_annual_revenue": 50000000, "description": "Cloud platform",
                }

                def score(messages):
                    payload = messages[-1]["content"]
                    supplied_offering, supplied_profile = payload.split("\n\nProspect profile:\n")
                    required = json.loads(supplied_offering.removeprefix("Offering:\n"))["required_tech_stack"]
                    stack = json.loads(supplied_profile)["tech_stack"]
                    self.assertEqual(stack, self.record["tech_stack"])
                    missing = [tech for tech in required if tech not in stack]
                    return agent.ProspectScore(
                        score=85 if missing else 100,
                        justification="Missing: " + ", ".join(missing) if missing else "All required technologies present.",
                        rubric_breakdown=agent.RubricBreakdown(
                            revenue_fit=100, tech_stack_match=85 if missing else 100,
                            segment_fit=100,
                        ),
                    )

                original = agent.build_prospect_profile.func("TEST-LEAD")["prospect_profile"]
                with patch.object(agent, "_scoring_llm") as scoring_llm:
                    scoring_llm.invoke.side_effect = score
                    before = agent.score_prospect.func(original, offering)
                    self.assertIn(technology, before["justification"])
                    agent.update_prospect_info.func("TEST-LEAD", technology)
                    refreshed = agent.build_prospect_profile.func("TEST-LEAD")["prospect_profile"]
                    self.assertIn(technology, refreshed["tech_stack"])
                    for profile in (original, refreshed):
                        after = agent.score_prospect.func(profile, offering)
                        self.assertEqual(after["score"], 100)
                        self.assertEqual(after["justification"], "All required technologies present.")
                        self.assertEqual(after["rubric_breakdown"]["tech_stack_match"], 100)


if __name__ == "__main__":
    unittest.main()
