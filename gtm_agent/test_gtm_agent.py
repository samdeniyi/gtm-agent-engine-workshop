import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    module = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ProspectToolsTest(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(module.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.prospect_id = next(iter(module.data_service.PROSPECTS))
        self.offering_id = next(iter(module.data_service.OFFERINGS))
        self.score = {"score": 75, "justification": "Good fit"}
        self.scorer = Mock()
        self.scorer.invoke.return_value.model_dump.return_value = self.score
        self.scoring = patch.object(module, "_scoring_llm", self.scorer)
        self.scoring.start()
        self.addCleanup(self.scoring.stop)

    def test_score_resolves_ids_and_builds_profile(self):
        result = module.score_prospect.invoke({
            "prospect_id": self.prospect_id, "offering_id": self.offering_id,
        })
        self.assertEqual(result, self.score)
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = content.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering), module.data_service.get_offering(self.offering_id))
        self.assertEqual(json.loads(profile), module.data_service._PROFILES[self.prospect_id])
        self.assertEqual(
            module.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
            {"prospect_profile": json.loads(profile), "found": True},
        )

    def test_missing_or_invalid_offering(self):
        for offering in (None, {}, {"description": "Incomplete"}):
            with self.subTest(offering=offering), patch.object(
                module.data_service, "get_offering", return_value=offering,
            ):
                self.assertEqual(
                    module.score_prospect.invoke({
                        "prospect_id": self.prospect_id, "offering_id": "invalid",
                    }),
                    {"score": None, "error": "Cannot score without a valid offering."},
                )
        self.scorer.invoke.assert_not_called()

    def test_cached_profile_uses_current_tech_stack_for_scoring(self):
        cached = {"prospect_id": self.prospect_id, "tech_stack": ["Stale"], "name": "Cached"}
        module.data_service._PROFILES[self.prospect_id] = cached
        module.score_prospect.invoke({
            "prospect_id": self.prospect_id, "offering_id": self.offering_id,
        })
        content = self.scorer.invoke.call_args.args[0][1]["content"]
        profile = json.loads(content.split("\n\nProspect profile:\n")[1])
        self.assertEqual(profile, {
            **cached, "tech_stack": module.data_service.fetch_tech_stack(self.prospect_id),
        })
        self.assertEqual(cached["tech_stack"], ["Stale"])

    def test_missing_prospect(self):
        self.assertEqual(
            module.score_prospect.invoke({
                "prospect_id": "missing", "offering_id": self.offering_id,
            }),
            {"score": None, "error": "Cannot score without a valid prospect."},
        )
        self.assertEqual(
            module.build_prospect_profile.invoke({"prospect_id": "missing"}),
            {"prospect_profile": None, "found": False},
        )
        self.scorer.invoke.assert_not_called()

    def test_scoring_and_email_graphs_without_rep_lookup(self):
        tools = [module.lookup_offering, module.build_prospect_profile, module.get_prospect,
                 module.send_prospect_email, module.score_prospect,
                 module.update_prospect_info, module.get_current_rep]
        self.assertNotIn("get_current_rep", module.SYSTEM_PROMPT)
        self.assertIn("get_current_rep", module.gtm_agent.nodes["tools"].bound.tools_by_name)
        for rep in module.REP_IDS[:2]:
            for name, args in (
                ("score_prospect", {"prospect_id": self.prospect_id, "offering_id": self.offering_id}),
                ("send_prospect_email", {
                    "prospect": module.data_service.get_prospect_record(self.prospect_id),
                    "subject": "Follow up", "body": "Hello",
                }),
            ):
                with self.subTest(tool=name, rep=rep["rep_id"]):
                    model = ToolCallingModel(messages=iter([
                        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}]),
                        AIMessage(content="Done"),
                    ]))
                    graph = module.create_deep_agent(
                        model=model, tools=tools, system_prompt=module.SYSTEM_PROMPT,
                    )
                    with patch.object(module, "gtm_agent", graph):
                        result = module.run_agent("Score or email this prospect", user_id=rep["rep_id"])
                    outputs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                    self.assertEqual([m.name for m in outputs], [name])
                    output = json.loads(outputs[0].content)
                    if name == "score_prospect":
                        self.assertEqual(output, self.score)
                    else:
                        self.assertEqual(output["status"], "sent")
                        self.assertEqual(output["from"], rep["email"])
                        self.assertEqual(output["from_name"], rep["name"])


if __name__ == "__main__":
    unittest.main()
