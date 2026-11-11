"""Minimal intent router and MCP client for the Medical Evidence API."""

import argparse
import asyncio
import re
import sys
from typing import Literal, Protocol

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, ListToolsResult
from pydantic import BaseModel

from app.models.answer import AskResponse
from app.models.evidence import SearchRequest, SearchResponse

MedicalTool = Literal["medical_search", "medical_ask"]

_MEDICAL_TERMS = re.compile(
    r"\b(?:medical|health|clinical|medicine|medicines|medication|medications|drug|drugs|"
    r"treatment|therapy|diagnosis|diagnose|symptom|symptoms|disease|condition|illness|"
    r"patient|dose|dosage|side effect|adverse event|guideline|clinical trial|"
    r"gerd|copd|hiv|aids|covid(?:-19)?|diabetes|hypertension|migraine|asthma|cancer|"
    r"pneumonia|influenza|hepatitis|ulcer|h\.?\s*pylori)\b|"
    r"پزشک|پزشکی|سلامت|درمان|دارو|علائم|بیماری|تشخیص|بالینی|شواهد|پژوهش|مقاله|راهنما|"
    r"عوارض|دوز|سرطان|عفونت|دیابت|فشار خون|سردرد",
    re.IGNORECASE,
)
_SEARCH_INTENT = re.compile(
    r"\b(?:evidence|literature|sources?|stud(?:y|ies)|research|guidelines?|papers?|"
    r"publications?|pubmed|latest|recent|newest|find|search|retrieve|systematic review)\b|"
    r"شواهد|منابع|مقاله|مقالات|پژوهش|جدیدترین|تازه|اخیر|راهنما|جستجو|پیدا کن",
    re.IGNORECASE,
)
_SYNTHESIS_INTENT = re.compile(
    r"\b(?:explain|interpret|compare|contrast|summari[sz]e|summary|difference between|"
    r"what does .+ mean|how does .+ work|implications?)\b|"
    r"توضیح|مقایسه|تفسیر|تفاوت|خلاصه|معنی",
    re.IGNORECASE,
)


class MCPToolClient(Protocol):
    async def list_tools(self) -> ListToolsResult: ...

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult: ...


class AgentResult(BaseModel):
    status: Literal["not_medical", "ok", "abstained", "tool_error"]
    tool_name: MedicalTool | None = None
    result: SearchResponse | AskResponse | None = None
    message: str | None = None


def select_medical_tool(question: str) -> MedicalTool | None:
    """Apply a small explicit policy; this function never generates an answer."""
    if not _MEDICAL_TERMS.search(question):
        return None
    if _SYNTHESIS_INTENT.search(question):
        return "medical_ask"
    if _SEARCH_INTENT.search(question):
        return "medical_search"
    return "medical_ask"


class MedicalAgent:
    def __init__(self, client: MCPToolClient):
        self.client = client

    async def respond(self, question: str, language: Literal["en", "fa"] = "en") -> AgentResult:
        tool_name = select_medical_tool(question)
        if tool_name is None:
            return AgentResult(
                status="not_medical",
                message="This agent handles medical evidence questions only.",
            )

        try:
            request = SearchRequest(query=question, language=language)
            tools = await self.client.list_tools()
            if tool_name not in {tool.name for tool in tools.tools}:
                return _tool_error(tool_name)

            response = await self.client.call_tool(
                tool_name,
                {"query": request.query, "language": request.language},
            )
            if response.isError or response.structuredContent is None:
                return _tool_error(tool_name)

            result_type = SearchResponse if tool_name == "medical_search" else AskResponse
            result = result_type.model_validate(response.structuredContent)
            return AgentResult(
                status="abstained" if result.abstained else "ok",
                tool_name=tool_name,
                result=result,
            )
        except Exception:
            # Keep provider or protocol failures from becoming an invented answer.
            return _tool_error(tool_name)


def _tool_error(tool_name: MedicalTool) -> AgentResult:
    return AgentResult(
        status="tool_error",
        tool_name=tool_name,
        message="The medical evidence tool failed; no medical answer was produced.",
    )


async def _run(question: str, language: Literal["en", "fa"]) -> AgentResult:
    if select_medical_tool(question) is None:
        return AgentResult(
            status="not_medical",
            message="This agent handles medical evidence questions only.",
        )

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await MedicalAgent(session).respond(question, language)


def main() -> None:
    parser = argparse.ArgumentParser(description="Route medical questions to the Medical Evidence MCP tools.")
    parser.add_argument("question", help="Medical or general question to route")
    parser.add_argument("--language", choices=("en", "fa"), default="en")
    args = parser.parse_args()
    response = asyncio.run(_run(args.question, args.language))
    print(response.model_dump_json())


if __name__ == "__main__":
    main()
