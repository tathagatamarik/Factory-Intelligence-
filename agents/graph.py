"""
LangGraph-based 3-agent chain for the HRMS Multi-Agent Assistant demo.

Agents:
    1. Intake Agent      - classifies the employee query.
    2. Resolver Agent    - answers leave_balance / policy_question queries using
                            a mock CSV dataset and a lightweight TF-IDF retriever
                            over a sample HR policy document.
    3. Escalation Agent  - handles grievance_or_disciplinary / other / ambiguous
                            queries by drafting an escalation note instead of
                            answering the substantive question.

The graph is compiled once and cached by the Streamlit layer via
@st.cache_resource (see tabs/hrms_agents.py).
"""

from __future__ import annotations

import os
import re
from typing import TypedDict, Optional, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Supported hosted-LLM providers for the demo. The UI (tabs/hrms_agents.py) lets
# the presenter pick a provider and model, and optionally paste in their own API
# key at runtime (overriding st.secrets) so the same demo can be re-run live
# against different providers/keys without editing files.
#
# Model choices are picked on a speed vs quality tradeoff and kept here as plain
# config rather than buried in node logic. Groq periodically retires older model
# IDs (e.g. the earlier llama-3.x model names used here were removed from Groq's
# catalog) — if a "model not found" error appears, check
# https://console.groq.com/docs/models for current model IDs and update below.
#   - Groq openai/gpt-oss-20b   -> faster, lighter quality
#   - Groq openai/gpt-oss-120b  -> slower, higher quality (default)
#   - OpenAI gpt-4o-mini        -> fast, low cost
#   - OpenAI gpt-4o             -> higher quality, slower/costlier
PROVIDER_CONFIG = {
    "groq": {
        "label": "Groq",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "default_model": os.environ.get("HRMS_GROQ_MODEL", "openai/gpt-oss-120b"),
        "secret_key": "GROQ_API_KEY",
    },
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "default_model": os.environ.get("HRMS_OPENAI_MODEL", "gpt-4o-mini"),
        "secret_key": "OPENAI_API_KEY",
    },
}

DEFAULT_PROVIDER = "groq"
# Backward-compatible alias used by earlier UI code / README references.
GROQ_MODEL_NAME = PROVIDER_CONFIG["groq"]["default_model"]

VALID_CATEGORIES = ["leave_balance", "policy_question", "grievance_or_disciplinary", "other"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAVE_CSV_PATH = os.path.join(BASE_DIR, "data", "mock_leave_balances.csv")
POLICY_TXT_PATH = os.path.join(BASE_DIR, "data", "sample_hr_policy.txt")

ESCALATION_NOTICE = (
    "This query has been routed to HR for review — the assistant does not resolve "
    "grievance or disciplinary matters."
)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    query: str
    provider: str
    api_key: Optional[str]
    model_name: Optional[str]
    employee_name: Optional[str]
    category: str
    classification_reasoning: str
    resolver_context: str
    response: str
    trace: List[dict]
    escalated: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# Lightweight HR policy retriever (TF-IDF, no external services required)
# ---------------------------------------------------------------------------

class PolicyRetriever:
    """Simple TF-IDF retriever over paragraph-level chunks of the policy doc."""

    def __init__(self, policy_path: str):
        with open(policy_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Split into paragraph-ish chunks on blank lines / section headers.
        chunks = [c.strip() for c in re.split(r"\n\s*\n", raw_text) if c.strip()]
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = [self.chunks[i] for i in top_indices if scores[i] > 0]
        return results if results else [self.chunks[0]]


def load_leave_balances() -> pd.DataFrame:
    return pd.read_csv(LEAVE_CSV_PATH)


def load_policy_retriever() -> PolicyRetriever:
    return PolicyRetriever(POLICY_TXT_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_employee(df: pd.DataFrame, query: str) -> Optional[pd.Series]:
    """Best-effort match of an employee name or ID mentioned in the query."""
    query_lower = query.lower()

    for _, row in df.iterrows():
        if str(row["employee_id"]).lower() in query_lower:
            return row

    for _, row in df.iterrows():
        name_parts = str(row["employee_name"]).lower().split()
        if any(part in query_lower for part in name_parts if len(part) > 2):
            return row

    return None


def _lookup_secret(secret_name: str) -> Optional[str]:
    try:
        import streamlit as st
        value = st.secrets.get(secret_name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(secret_name)


def resolve_api_key(provider: str, override_key: Optional[str]) -> Optional[str]:
    """Returns the effective API key for a provider: an explicit override (e.g.
    typed into the UI) takes precedence, falling back to st.secrets/env."""
    if override_key:
        return override_key
    config = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG[DEFAULT_PROVIDER])
    return _lookup_secret(config["secret_key"])


def _build_llm(state: AgentState):
    provider = state.get("provider") or DEFAULT_PROVIDER
    config = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG[DEFAULT_PROVIDER])
    model_name = state.get("model_name") or config["default_model"]

    api_key = resolve_api_key(provider, state.get("api_key"))
    if not api_key:
        raise RuntimeError(
            f"{config['secret_key']} is not configured. Enter an API key in the UI, or add "
            f"it to .streamlit/secrets.toml (see .streamlit/secrets.toml.example) before "
            f"using the HRMS assistant."
        )

    if provider == "openai":
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.1)

    # default / "groq"
    return ChatGroq(model=model_name, api_key=api_key, temperature=0.1)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def intake_node(state: AgentState) -> AgentState:
    """Classifies the incoming query into one of VALID_CATEGORIES."""
    query = state["query"]
    trace = list(state.get("trace", []))

    llm = _build_llm(state)
    system_prompt = (
        "You are an HR query intake classifier for a manufacturing company. "
        "Classify the employee's message into exactly one of these categories:\n"
        "- leave_balance: questions about how much leave/WFH balance the employee has.\n"
        "- policy_question: questions about company policy (leave rules, WFH rules, "
        "reimbursement rules, etc.) that do not require personal balance lookup.\n"
        "- grievance_or_disciplinary: complaints, harassment, conflict with a colleague or "
        "manager, disciplinary action, show-cause, or any sensitive/ambiguous personal matter.\n"
        "- other: anything that does not clearly fit the above.\n\n"
        "Respond with ONLY the category label, nothing else."
    )

    try:
        result = llm.invoke(
            [
                ("system", system_prompt),
                ("human", query),
            ]
        )
        raw_label = result.content.strip().lower()
    except Exception as exc:
        raise RuntimeError(f"llm_api_error: {exc}") from exc

    category = next((c for c in VALID_CATEGORIES if c in raw_label), "other")

    trace.append(
        {
            "agent": "Intake Agent",
            "action": "Classified query",
            "detail": f'Category selected: "{category}"',
        }
    )

    return {
        **state,
        "category": category,
        "classification_reasoning": raw_label,
        "trace": trace,
    }


def resolver_node(state: AgentState) -> AgentState:
    """Answers leave_balance / policy_question queries using the mock dataset
    and the TF-IDF policy retriever."""
    query = state["query"]
    category = state["category"]
    trace = list(state.get("trace", []))

    df = load_leave_balances()
    retriever = load_policy_retriever()

    context_parts = []

    if category == "leave_balance":
        employee_row = _find_employee(df, query)
        if employee_row is not None:
            context_parts.append(
                "Employee leave record found:\n"
                f"- Name: {employee_row['employee_name']} ({employee_row['employee_id']})\n"
                f"- Department: {employee_row['department']}\n"
                f"- Casual Leave balance: {employee_row['casual_leave_balance']} days\n"
                f"- Sick Leave balance: {employee_row['sick_leave_balance']} days\n"
                f"- Earned Leave balance: {employee_row['earned_leave_balance']} days\n"
                f"- Work From Home balance: {employee_row['work_from_home_balance']} days"
            )
            trace.append(
                {
                    "agent": "Resolver Agent",
                    "action": "Looked up leave balance record",
                    "detail": f"Matched employee: {employee_row['employee_name']} ({employee_row['employee_id']})",
                }
            )
        else:
            context_parts.append(
                "No specific employee could be identified from the query in the mock "
                "leave-balance dataset. Ask the employee to confirm their employee ID or "
                "name if a personal balance lookup is needed."
            )
            trace.append(
                {
                    "agent": "Resolver Agent",
                    "action": "Looked up leave balance record",
                    "detail": "No matching employee found in mock dataset — answering generically.",
                }
            )

    if category in ("policy_question", "leave_balance"):
        policy_chunks = retriever.retrieve(query, top_k=2)
        context_parts.append(
            "Relevant HR policy excerpt(s):\n" + "\n---\n".join(policy_chunks)
        )
        trace.append(
            {
                "agent": "Resolver Agent",
                "action": "Retrieved relevant policy sections (TF-IDF search)",
                "detail": f"Retrieved {len(policy_chunks)} matching passage(s) from the sample HR policy document.",
            }
        )

    context = "\n\n".join(context_parts)

    llm = _build_llm(state)
    system_prompt = (
        "You are the HR Resolver Agent for a cement manufacturing company's HR assistant. "
        "Answer the employee's question clearly, briefly and professionally using ONLY the "
        "context provided below. If the context does not fully answer the question, say so "
        "honestly and suggest the employee confirm with HR. Do not invent numbers or policy "
        "rules that are not in the context. Keep the answer concise (3-6 sentences) and do not "
        "use emojis.\n\nCONTEXT:\n" + context
    )

    try:
        result = llm.invoke(
            [
                ("system", system_prompt),
                ("human", query),
            ]
        )
        answer = result.content.strip()
    except Exception as exc:
        raise RuntimeError(f"llm_api_error: {exc}") from exc

    trace.append(
        {
            "agent": "Resolver Agent",
            "action": "Generated final response",
            "detail": "Answer composed from mock leave data and/or retrieved policy context.",
        }
    )

    return {
        **state,
        "resolver_context": context,
        "response": answer,
        "trace": trace,
        "escalated": False,
    }


def escalation_node(state: AgentState) -> AgentState:
    """Handles grievance_or_disciplinary and other/ambiguous queries by drafting
    an escalation note instead of answering the substantive question."""
    category = state["category"]
    trace = list(state.get("trace", []))

    if category == "grievance_or_disciplinary":
        reason = "the query concerns a grievance or disciplinary matter"
    else:
        reason = "the query did not clearly match an automated resolution category"

    note = (
        f"{ESCALATION_NOTICE}\n\n"
        f"Reason for escalation: {reason}. A member of the HR team will follow up directly "
        f"with the employee. No automated response has been generated for the substantive "
        f"question in order to ensure this matter receives proper human review."
    )

    trace.append(
        {
            "agent": "Escalation Agent",
            "action": "Drafted escalation note and logged the query as escalated",
            "detail": f"Escalation reason: {reason}.",
        }
    )

    return {
        **state,
        "response": note,
        "trace": trace,
        "escalated": True,
    }


def route_after_intake(state: AgentState) -> str:
    if state["category"] in ("leave_balance", "policy_question"):
        return "resolver"
    return "escalation"


# ---------------------------------------------------------------------------
# Graph construction (compiled once, cached by Streamlit layer)
# ---------------------------------------------------------------------------

def build_hrms_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("escalation", escalation_node)

    graph.set_entry_point("intake")
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {"resolver": "resolver", "escalation": "escalation"},
    )
    graph.add_edge("resolver", END)
    graph.add_edge("escalation", END)

    return graph.compile()


def run_hrms_query(
    compiled_graph,
    query: str,
    provider: str = DEFAULT_PROVIDER,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> AgentState:
    """Runs a single query through the compiled graph and returns the final state.

    provider/api_key/model_name are threaded into the initial state so each node
    builds its LLM client against whichever provider and key the UI selected for
    this run, without needing to recompile the graph.
    """
    initial_state: AgentState = {
        "query": query,
        "provider": provider,
        "api_key": api_key,
        "model_name": model_name,
        "trace": [],
    }
    final_state = compiled_graph.invoke(initial_state)
    return final_state
