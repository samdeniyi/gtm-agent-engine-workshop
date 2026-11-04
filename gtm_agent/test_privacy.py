import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import tracing_context

from . import data_service

with patch("dotenv.load_dotenv"), patch("langchain_openai.ChatOpenAI"), \
        patch("deepagents.create_deep_agent"):
    agent = importlib.import_module(".gtm_agent", __package__)


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.pid = next(iter(data_service.PROSPECTS))

    def assert_no_billing(self, payload):
        serialized = json.dumps(payload)
        for field in ("billing_qualification", "card_on_file", "tax_id"):
            self.assertNotIn(field, serialized)

    def test_contact_allowlist_and_email_arguments(self):
        record = dict(data_service.PROSPECTS[self.pid], prospect_id="wrong", extra="private")
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = agent.get_prospect.invoke({"prospect_id": self.pid})
        contact = result["prospect"]
        self.assertTrue(result["found"])
        self.assertEqual(set(contact), {"prospect_id", "name", "email", "disqualified",
                                        "annual_revenue", "enrichment_source"})
        self.assertEqual(contact["prospect_id"], self.pid)
        arguments = {"prospect": contact, "subject": "Hello", "body": "Let's talk"}
        self.assert_no_billing(arguments)
        sent = agent.send_prospect_email.func(
            **arguments, runtime=SimpleNamespace(config={}),
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], contact["email"])
        self.assert_no_billing(result)
        self.assert_no_billing(sent)

    def test_new_profile_is_safe_before_save_and_on_cache_hit(self):
        saved = []
        original_save = data_service.save_profile_to_db

        def save(prospect_id, profile):
            saved.append(profile)
            return original_save(prospect_id, profile)

        with patch.object(data_service, "save_profile_to_db", side_effect=save):
            result = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        self.assertTrue(result["found"])
        self.assert_no_billing(saved)
        self.assert_no_billing(data_service._PROFILES)
        self.assert_no_billing(result)
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.pid}), result)
        for field in ("engagement_history", "account_details", "tech_stack"):
            self.assertEqual(result["prospect_profile"][field], data_service.PROSPECTS[self.pid][field])

    def test_legacy_cache_trace_output_is_safe(self):
        data_service._PROFILES[self.pid] = dict(data_service.PROSPECTS[self.pid], prospect_id=self.pid)
        runs = []
        with tracing_context(enabled="local"):
            result = data_service.get_profile_from_db(
                self.pid, langsmith_extra={"on_end": runs.append},
            )
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].outputs, result)
        self.assert_no_billing(runs[0].outputs)
        self.assert_no_billing(data_service._PROFILES)
        self.assert_no_billing(agent.build_prospect_profile.invoke({"prospect_id": self.pid}))

    def test_scoring_allowlist_and_tool_arguments(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.pid})["prospect_profile"]
        offering = next(iter(data_service.OFFERINGS.values()))
        arguments = {"prospect_profile": profile, "offering": offering}
        self.assert_no_billing(arguments)
        incoming = dict(profile, billing_qualification={"tax_id": "private"},
                        extra="private", tech_stack=["stale"])
        model = Mock()
        model.invoke.return_value.model_dump.return_value = {"score": 75}
        with patch.object(agent, "_scoring_llm", model):
            result = agent.score_prospect.invoke({**arguments, "prospect_profile": incoming})
        self.assertEqual(result, {"score": 75})
        messages = model.invoke.call_args.args[0]
        scored = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        fields = {"prospect_id", "name", "annual_revenue", "tech_stack",
                  "account_details", "engagement_history"}
        self.assertEqual(scored, {k: profile[k] for k in fields})
        self.assert_no_billing(messages)
        self.assert_no_billing(result)

    def test_missing_records_and_invalid_offering(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})
        with patch.object(agent, "_scoring_llm") as model:
            for offering in (None, {}):
                self.assertIsNone(agent.score_prospect.invoke(
                    {"prospect_profile": {}, "offering": offering},
                )["score"])
            model.invoke.assert_not_called()
