from typing import TypedDict
from pydantic import BaseModel, Field
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# -------- Structured plan --------
class RetrievalPlan(BaseModel):
    """Structured decision by planner agent."""
    source: Literal["local", "web", "history"] = Field(
        description="Choose 'local' for vector DB (PDF knowledge), 'web' for Serper internet search, "
                    "or 'history' to rely on conversation memory."
    )
    query: str = Field(description="Refined query to use with chosen retriever.")


class Planner:
    """Planner LLM → decides which retriever to use (local, web, or memory)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        # structured output ensures we only get JSON with {source, query}
        self.llm = ChatOpenAI(model=model, temperature=0).with_structured_output(RetrievalPlan)

    def decide(self, question: str) -> RetrievalPlan:
        system = (
            "You are a planner that decides where to search for information.\n"
            "- If the user asks about EU treaties, laws, or static PDF info, use 'local'.\n"
            "- If the user asks about news, recent events, or general updates, use 'web'.\n"
            "- If the question can be answered using prior conversation context, use 'history'.\n"
            "Always return JSON with {source, query}."
        )
        return self.llm.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": question}]
        )


# -------- LangGraph state --------
class AgentState(TypedDict):
    question: str
    plan: dict


class AgentWorkflow:
    """
    LangGraph workflow:
      - plan → produce structured output
      - end
    """

    def __init__(self):
        self.planner = Planner()

        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self._plan)
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", END)

        self.app = workflow.compile()

    def _plan(self, state: AgentState) -> AgentState:
        decision = self.planner.decide(state["question"])
        return {"plan": decision.dict()}

    def run(self, question: str) -> dict:
        result = self.app.invoke({"question": question})
        return result["plan"]
