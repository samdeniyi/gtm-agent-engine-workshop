import importlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

module_dir = Path(__file__).resolve().parent
sys.path[:] = [path for path in sys.path if Path(path).resolve() != module_dir]
sys.path.insert(0, str(module_dir.parent))
with patch("dotenv.load_dotenv"), patch.dict(
    os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}
):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class AgentTests(unittest.TestCase):
    def exercise_request(self, request, steps, user_id, intent):
        calls = []

        def respond(model, messages, **kwargs):
            calls.append((model, messages, kwargs))
            response = steps[len(calls) - 1] if len(calls) <= len(steps) else AIMessage(content="Done")
            return ChatResult(generations=[ChatGeneration(message=response)])

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), patch.object(
            ChatOpenAI, "_generate", respond
        ), patch.object(agent.gtm_agent, "invoke", wraps=agent.gtm_agent.invoke) as invoke:
            result = agent.run_agent(
                request, user_id=user_id, environment="test", thread_id="crm-thread"
            )

        config = invoke.call_args.kwargs["config"]
        self.assertEqual(config["run_name"], "GTM Assistant")
        self.assertIsInstance(config["run_id"], uuid.UUID)
        self.assertEqual(result["run_id"], str(config["run_id"]))
        self.assertEqual(config["metadata"], {
            "thread_id": "crm-thread", "user_id": user_id,
            "environment": "test", "request_intent": intent,
        })
        self.assertEqual(result["reply"], "Done")
        self.assertEqual(len(calls), len(steps) + 1)
        expected_tools = [
            agent.lookup_offering, agent.build_prospect_profile, agent.get_prospect,
            agent.send_prospect_email, agent.score_prospect,
            agent.update_prospect_info, agent.get_current_rep,
        ]
        for model, messages, kwargs in calls:
            self.assertIs(model, agent.agent_model)
            self.assertEqual(model.model_name, agent.MODEL_NAME)
            self.assertEqual(model.temperature, 0)
            self.assertEqual(
                [message.content for message in messages if isinstance(message, SystemMessage)],
                [agent.SYSTEM_PROMPT],
            )
            schemas = kwargs["tools"]
            self.assertEqual(
                [schema["function"]["name"] for schema in schemas],
                [tool.name for tool in expected_tools],
            )
            for schema, tool in zip(schemas, expected_tools):
                self.assertEqual(schema["function"]["description"], tool.description)
                self.assertEqual(
                    schema["function"]["parameters"]["properties"],
                    convert_to_openai_tool(tool)["function"]["parameters"]["properties"],
                )
                self.assertNotIn("runtime", schema["function"]["parameters"]["properties"])
        self.assertEqual(len(calls[0][1]), 2)
        self.assertEqual(calls[0][1][1].content, request)
        return {
            message.name: json.loads(message.content)
            for message in result["messages"] if isinstance(message, ToolMessage)
        }

    def tool_call(self, name, **args):
        return AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call",
        }])

    def test_email_flow_injects_signed_in_rep_without_harness(self):
        prospect_id, prospect = next(iter(agent.data_service.PROSPECTS.items()))
        for rep in agent.REP_IDS[:2]:
            with self.subTest(rep=rep["rep_id"]):
                outputs = self.exercise_request(
                    "Email the prospect a follow up",
                    [
                        self.tool_call("get_current_rep"),
                        self.tool_call("get_prospect", prospect_id=prospect_id),
                        self.tool_call("send_prospect_email", prospect={
                            "name": prospect["name"], "email": prospect["email"],
                        }, subject="Follow up", body="Thanks for your time."),
                    ], rep["rep_id"], "send_email",
                )
                self.assertEqual(outputs["get_current_rep"], {"rep": rep, "found": True})
                self.assertEqual(outputs["get_prospect"]["prospect"]["email"], prospect["email"])
                sent = outputs["send_prospect_email"]
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["from"], rep["email"])
                self.assertEqual(sent["from_name"], rep["name"])
                self.assertEqual(sent["to"], prospect["email"])
                self.assertEqual(sent["subject"], "Follow up")
                self.assertEqual(sent["body"], "Thanks for your time.")

    def test_profile_scoring_and_update_flow(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        offering_id, offering = next(iter(agent.data_service.OFFERINGS.items()))
        profile = {"prospect_id": prospect_id}
        score = agent.ProspectScore(
            score=75, justification="Suitable fit", rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=70, segment_fit=75,
            ),
        )
        with patch.object(agent, "_scoring_llm") as scoring_llm, patch.dict(
            agent.data_service._PROFILES, {}, clear=True
        ):
            scoring_llm.invoke.return_value = score
            outputs = self.exercise_request(
                "Update the prospect and score their fit",
                [
                    self.tool_call("get_current_rep"),
                    self.tool_call("lookup_offering", offering_id=offering_id),
                    self.tool_call("build_prospect_profile", prospect_id=prospect_id),
                    self.tool_call("score_prospect", prospect_profile=profile, offering=offering),
                    self.tool_call("update_prospect_info", prospect_id=prospect_id, technology="Python"),
                ], agent.REP_IDS[0]["rep_id"], "update_prospect",
            )
        self.assertEqual(outputs["lookup_offering"]["offering"], offering)
        self.assertEqual(outputs["build_prospect_profile"]["prospect_profile"]["prospect_id"], prospect_id)
        self.assertEqual(outputs["score_prospect"], score.model_dump())
        self.assertTrue(outputs["update_prospect_info"]["updated"])
        self.assertIn("Python", outputs["update_prospect_info"]["tech_stack"])


if __name__ == "__main__":
    unittest.main()
