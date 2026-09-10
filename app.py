"""
Main entry point for the client demo Streamlit application.

Two tabs:
    1. VisionAI - Plant Safety      (tabs/vision_ppe.py)
    2. HRMS Multi-Agent Assistant   (tabs/hrms_agents.py)
"""

import streamlit as st

from tabs import vision_ppe, hrms_agents

st.set_page_config(
    page_title="Plant Intelligence Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    /* Hide default Streamlit chrome: hamburger menu, footer, "Made with Streamlit" badge */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    a[href*="streamlit.io/cloud"] {display: none !important;}
    .stAppDeployButton {display: none !important;}

    /* White background, clean sans-serif typography */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF;
    }
    * {
        font-family: "Source Sans Pro", "Helvetica Neue", Arial, sans-serif;
    }

    :root {
        --accent-color: #1A3A6B;
        --accent-color-light: #EAF0FA;
    }

    /* Accent color for primary buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: var(--accent-color);
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #142c53;
        color: #FFFFFF;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #EEEEEE;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        white-space: pre-wrap;
        font-weight: 600;
        color: #333333;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-color) !important;
        border-bottom: 3px solid var(--accent-color) !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #14213D;
        font-weight: 700;
    }

    /* Chat input accent */
    [data-testid="stChatInput"] {
        border-color: var(--accent-color);
    }

    /* Agent trace styling */
    .trace-summary {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        padding: 12px 16px;
        background-color: var(--accent-color-light);
        border-radius: 8px;
    }
    .trace-summary-label {
        font-size: 0.85rem;
        color: #555555;
        font-weight: 600;
    }
    .trace-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        background-color: var(--accent-color);
        color: #FFFFFF;
    }
    .trace-pill-resolved {
        background-color: #2E7D32;
    }
    .trace-pill-escalated {
        background-color: #B23B3B;
    }

    /* Reduce cramped top padding */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Plant Intelligence Demo")
st.caption("AI agent use cases for cement manufacturing operations")

tab_vision, tab_hrms = st.tabs(["VisionAI — Plant Safety", "HRMS Multi-Agent Assistant"])

with tab_vision:
    vision_ppe.render()

with tab_hrms:
    hrms_agents.render()
