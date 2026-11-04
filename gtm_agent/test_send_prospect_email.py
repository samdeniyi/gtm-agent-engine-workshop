import importlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.prospect = {"prospect_id": "TEST-LEAD", "name": "Test Prospect",
                         "email": "supplied@example.com", "disqualified": False}
        self.record = {**self.prospect, "email": "source@example.com"}
        self.records = patch.dict(agent.data_service.PROSPECTS, {"TEST-LEAD": self.record})
        self.records.start()
        self.addCleanup(self.records.stop)

    def send(self):
        return agent.send_prospect_email.func(
            self.prospect, "Hello", "Following up", self.runtime,
        )

    def test_disqualified_source_blocks_send_despite_supplied_eligibility(self):
        self.record["disqualified"] = True
        with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Send reached")):
            self.assertEqual(self.send(), {
                "status": "blocked",
                "error": "Prospect is disqualified; email not sent.",
                "prospect_id": "TEST-LEAD",
            })

    def test_eligible_send_uses_source_email(self):
        result = self.send()
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "source@example.com")
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Following up")
        self.assertTrue(result["message_id"].startswith("msg-"))

    def test_unknown_source_fails_without_sending(self):
        self.prospect["prospect_id"] = "MISSING-LEAD"
        with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Send reached")):
            self.assertEqual(self.send(), {
                "status": "failed", "error": "Prospect record not found.",
            })

    def test_missing_id_fails_without_sending(self):
        self.prospect.pop("prospect_id")
        with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Send reached")):
            self.assertEqual(self.send()["status"], "failed")

    def test_missing_source_email_cannot_use_supplied_email(self):
        self.record.pop("email")
        with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Send reached")):
            self.assertEqual(self.send(), {
                "status": "failed", "error": "Prospect record has no email address.",
            })


@unittest.skipUnless(HAS_MODEL_KEY, "Live agent checks require OPENAI_API_KEY")
class EmailAgentTests(unittest.TestCase):
    def test_disqualified_send_is_reported_to_rep(self):
        result = agent.run_agent(
            "Email prospect LEAD-50001 with subject Hello and body Following up.",
            user_id="rep_amills",
        )
        tool_results = [message.content for message in result["messages"]
                        if getattr(message, "name", None) == "send_prospect_email"]
        self.assertTrue(any('"blocked"' in content for content in tool_results))
        reply = result["reply"].lower()
        self.assertIn("disqualified", reply)
        self.assertTrue(any(text in reply for text in ("not sent", "did not send", "wasn't sent")))

    def test_eligible_send_is_reported_to_rep(self):
        result = agent.run_agent(
            "Email prospect LEAD-90001 with subject Hello and body Following up.",
            user_id="rep_amills",
        )
        tool_results = [message.content for message in result["messages"]
                        if getattr(message, "name", None) == "send_prospect_email"]
        self.assertTrue(any('"sent"' in content for content in tool_results))
        self.assertIn("sent", result["reply"].lower())


if __name__ == "__main__":
    unittest.main()
