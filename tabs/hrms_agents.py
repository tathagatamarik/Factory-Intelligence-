"""
Tab 2: HRMS Multi-Agent Assistant.

Demonstrates a 3-agent LangGraph chain (Intake -> Resolver / Escalation) answering
employee HR queries, calling an open-weight/hosted model via a provider chosen at
runtime (Groq or OpenAI), using langchain-groq / langchain-openai respectively.
"""

import streamlit as st

from agents.graph import (
    build_hrms_graph,
    run_hrms_query,
    resolve_api_key,
    PROVIDER_CONFIG,
    DEFAULT_PROVIDER,
    VALID_CATEGORIES,
)

CATEGORY_LABELS = {
    "leave_balance": "Leave Balance",
    "policy_question": "Policy Question",
    "grievance_or_disciplinary": "Grievance / Disciplinary",
    "other": "Other",
}

EXAMPLE_QUERIES = [
    "How many casual leave days does E1003 have left?",
    "What is the company's work from home policy for plant employees?",
    "How do I submit a travel reimbursement claim?",
    "I want to file a complaint against my manager for unfair treatment.",
]


@st.cache_resource(show_spinner=False)
def get_compiled_graph():
    """Compiles the LangGraph agent chain once and reuses it across reruns.

    The compiled graph does not depend on which provider/key/model is chosen —
    that is threaded in per-call via run_hrms_query, so a single compiled graph
    can be reused across provider switches within the same session.
    """
    return build_hrms_graph()


def _render_provider_settings():
    """Renders the provider / API key / model selection panel and returns
    (provider, api_key, model_name, is_ready)."""
    with st.expander("Model provider settings", expanded=True):
        st.caption(
            "Choose which hosted LLM provider powers the agent chain. Paste an API "
            "key here to test with a different key than the one configured in "
            "secrets — it is used only for this session and is not stored."
        )

        provider_options = list(PROVIDER_CONFIG.keys())
        default_index = provider_options.index(DEFAULT_PROVIDER)
        provider = st.selectbox(
            "Provider",
            options=provider_options,
            index=default_index,
            format_func=lambda p: PROVIDER_CONFIG[p]["label"],
            key="hrms_provider_select",
        )
        config = PROVIDER_CONFIG[provider]

        col_a, col_b = st.columns(2)
        with col_a:
            model_name = st.selectbox(
                "Model",
                options=config["models"],
                index=0,
                key=f"hrms_model_select_{provider}",
            )
        with col_b:
            key_input = st.text_input(
                f"{config['label']} API key (optional)",
                type="password",
                placeholder=f"Overrides {config['secret_key']} from secrets.toml",
                key=f"hrms_key_input_{provider}",
            )

        effective_key = resolve_api_key(provider, key_input.strip() if key_input else None)
        using_own_key = bool(key_input.strip()) if key_input else False

        if effective_key:
            source = "key entered above" if using_own_key else f"{config['secret_key']} in secrets.toml"
            st.caption(f"Using {config['label']} ({model_name}) — key source: {source}.")
        else:
            st.warning(
                f"No {config['label']} API key found. Enter one above, or add "
                f"{config['secret_key']} to .streamlit/secrets.toml, to use this provider."
            )

    return provider, effective_key, model_name, bool(effective_key)


def _render_trace(trace: list, category: str, escalated: bool):
    st.markdown("#### Agent Trace")
    st.caption(
        "Step-by-step record of which agent handled this query and why."
    )

    for i, step in enumerate(trace, start=1):
        with st.container(border=True):
            cols = st.columns([0.08, 0.92])
            with cols[0]:
                st.markdown(f"**{i}**")
            with cols[1]:
                st.markdown(f"**{step['agent']}** — {step['action']}")
                st.caption(step["detail"])

    badge_text = "Escalated to HR" if escalated else "Resolved automatically"
    st.markdown(
        f"""
        <div class="trace-summary">
            <span class="trace-summary-label">Final classification:</span>
            <span class="trace-pill">{CATEGORY_LABELS.get(category, category)}</span>
            <span class="trace-summary-label">Outcome:</span>
            <span class="trace-pill {'trace-pill-escalated' if escalated else 'trace-pill-resolved'}">{badge_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.markdown("### HRMS Multi-Agent Assistant")
    st.write(
        "A 3-agent assistant for employee HR queries: an Intake Agent classifies the "
        "question, a Resolver Agent answers leave-balance and policy questions from "
        "reference data, and an Escalation Agent routes grievance or disciplinary "
        "matters directly to human HR review instead of answering them automatically."
    )

    provider, api_key, model_name, is_ready = _render_provider_settings()

    if not is_ready:
        st.info(
            "You can still type a question below — configure an API key above "
            "before submitting it to get a response."
        )

    with st.expander("Try an example query", expanded=False):
        example_cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_QUERIES):
            with example_cols[i % 2]:
                if st.button(example, key=f"example_{i}", width="stretch"):
                    st.session_state["hrms_query_input"] = example

    query = st.chat_input("Ask an HR question (e.g. leave balance, WFH policy, reimbursement)...")

    if "hrms_query_input" in st.session_state and not query:
        query = st.session_state.pop("hrms_query_input")

    if "hrms_history" not in st.session_state:
        st.session_state["hrms_history"] = []

    if query:
        try:
            with st.spinner("Routing query through the agent chain..."):
                compiled_graph = get_compiled_graph()
                final_state = run_hrms_query(
                    compiled_graph, query, provider=provider, api_key=api_key, model_name=model_name
                )
            st.session_state["hrms_history"].append(
                {"query": query, "state": final_state, "error": None, "provider": provider, "model_name": model_name}
            )
        except Exception as exc:
            error_message = _format_error(exc, provider)
            st.session_state["hrms_history"].append(
                {"query": query, "state": None, "error": error_message, "provider": provider, "model_name": model_name}
            )

    if not st.session_state["hrms_history"]:
        st.info("Submit a query above to see the assistant respond.")
        return

    for entry in reversed(st.session_state["hrms_history"]):
        st.markdown("---")
        st.markdown(f"**Employee query:** {entry['query']}")

        if entry["error"]:
            st.error(entry["error"])
            continue

        state = entry["state"]
        st.markdown("**Assistant response:**")
        with st.container(border=True):
            st.write(state["response"])

        with st.expander("View Agent Trace", expanded=True):
            _render_trace(state.get("trace", []), state.get("category", "other"), state.get("escalated", False))

        st.caption(
            f"Model used: {PROVIDER_CONFIG[entry['provider']]['label']} — {entry['model_name']}"
        )


def _format_error(exc: Exception, provider: str) -> str:
    message = str(exc)
    provider_label = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG[DEFAULT_PROVIDER])["label"]
    secret_key = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG[DEFAULT_PROVIDER])["secret_key"]

    if "is not configured" in message:
        return (
            f"{secret_key} is not configured. Enter an API key in the model provider "
            f"settings above, or add it to .streamlit/secrets.toml, before using the "
            f"HRMS assistant."
        )

    lowered = message.lower()
    if "rate limit" in lowered or "429" in message:
        return (
            f"The {provider_label} API rate limit has been reached. Please wait a "
            f"moment and try the query again."
        )
    if "unauthorized" in lowered or "401" in message or "invalid api key" in lowered or "incorrect api key" in lowered:
        return (
            f"The {provider_label} API rejected the configured API key. Please verify "
            f"the key entered in the model provider settings above."
        )
    if "model_not_found" in lowered or "does not exist" in lowered or "404" in message:
        return (
            f"The {provider_label} API rejected the configured model name (it may have "
            f"been retired or renamed by the provider). Please check the current model "
            f"IDs in the {provider_label} console/docs and update the model configuration."
        )
    if "llm_api_error" in message:
        return (
            f"The assistant could not reach the {provider_label} API right now. Please "
            f"try again shortly. If the problem persists, verify your network "
            f"connection and API key configuration."
        )

    return (
        "An unexpected error occurred while processing this query. Please try "
        "again. If the problem persists, contact the system administrator."
    )
