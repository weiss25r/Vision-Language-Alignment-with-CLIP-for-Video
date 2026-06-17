import streamlit as st
import pandas as pd
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference.pipeline import InferencePipeline, InferenceModel

pipeline = InferencePipeline(
    model_path="./app/model.pth",
    video_embeddings_path="./app/video_embeddings_test_seen.pt",
    df_path="./data/annotations/processed/test_seen.csv"
)

st.set_page_config(
    page_title="VideoCLIP Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #111; color: #e0e0e0; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Search bar styling */
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 6px;
        font-size: 1rem;
        padding: 0.5rem 0.75rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6e9fff;
        box-shadow: 0 0 0 2px rgba(110,159,255,0.25);
    }

    /* Button */
    .stButton > button {
        background-color: #2a2a2a;
        color: #e0e0e0;
        border: 1px solid #555;
        border-radius: 6px;
        width: 100%;
        padding: 0.5rem;
    }
    .stButton > button:hover {
        border-color: #6e9fff;
        color: #6e9fff;
    }

    /* Dataframe table */
    .stDataFrame { border: 1px solid #2a2a2a; border-radius: 6px; overflow: hidden; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }

    /* Back button area */
    .back-link { color: #6e9fff; font-size: 0.85rem; cursor: pointer; }

    /* Align search button to bottom of its column (matches input baseline) */
    div[data-testid="stHorizontalBlock"]:has([data-testid="search_btn"]) > div {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }

    /* Search icon button */
    button[data-testid="baseButton-secondary"]:has(p:empty) {
        /* fallback — target by key below */
    }
    [data-testid="search_btn"] > button,
    #search_btn {
        background-color: #1e1e1e !important;
        border: 1px solid #444 !important;
        border-radius: 6px !important;
        color: #e0e0e0 !important;
        font-size: 1.1rem !important;
        padding: 0.35rem 0.75rem !important;
        width: 100%;
    }
    [data-testid="search_btn"] > button:hover,
    #search_btn:hover {
        border-color: #6e9fff !important;
        color: #6e9fff !important;
    }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "search"
if "query" not in st.session_state:
    st.session_state.query = ""
if "results" not in st.session_state:
    st.session_state.results = None

def run_retrieval(query: str) -> pd.DataFrame:
    df = pipeline.run(query)
    return df

def page_search():
    # Push content down to vertical center
    st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)

    _, center, _ = st.columns([2, 3, 2])
    with center:
        st.markdown(
            "<div style='text-align:center;'>"
            "<p style='font-size:2.6rem; font-weight:700; color:#e0e0e0; "
            "margin-bottom:0.15rem; letter-spacing:-0.5px;'>VideoCLIP</p>"
            "<p style='font-size:0.9rem; color:#666; margin-bottom:1.8rem;'>"
            "Text-to-video retrieval · Epic-Kitchens dataset</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        inp_col, btn_col = st.columns([11, 1])
        with inp_col:
            query = st.text_input(
                label="Query",
                placeholder="e.g. cut the onion into slices",
                label_visibility="collapsed",
                key="search_input",
            )
        with btn_col:
            search_clicked = st.button("🔍", use_container_width=True, key="search_btn")

        # Custom javascript to trigger search on Enter key

        st.iframe("""
        <script>
        (function() {
            function attach() {
                const input = window.parent.document.querySelector('input[placeholder="e.g. cut the onion into slices"]');
                if (!input) { setTimeout(attach, 100); return; }
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        e.stopPropagation();
                        // blur commits the current value to Streamlit's state
                        input.blur();
                        // small delay to let Streamlit process the blur before the click
                        setTimeout(function() {
                            const btns = window.parent.document.querySelectorAll('button');
                            for (const btn of btns) {
                                if (btn.innerText.trim() === '🔍') { btn.click(); break; }
                            }
                        }, 50);
                    }
                });
            }
            attach();
        })();
        </script>
        """)

    if search_clicked:
        q = query.strip()
        if not q:
            st.warning("Please enter a query before searching.")
        else:
            with st.spinner("Running retrieval…"):
                results = run_retrieval(q)
            st.session_state.query = q
            st.session_state.results = results
            st.session_state.page = "results"
            st.rerun()

def page_results():
    if st.session_state.results is None:
        st.session_state.page = "search"
        st.rerun()
        return

    df = st.session_state.results
    query = st.session_state.query

    if st.button("← New search"):
        st.session_state.page = "search"
        st.session_state.results = None
        st.rerun()

    st.markdown(f"### Results for: `{query}`")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clips retrieved", len(df))
    c2.metric("Unique videos", df["video_id"].nunique())
    c3.metric("Avg clip length (frames)",
              int((df["stop_frame"] - df["start_frame"]).mean()))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results table ─────────────────────────────────────────────────────────
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Video ID":     st.column_config.TextColumn("Video ID",     width="small"),
            "Narration ID": st.column_config.TextColumn("Narration ID", width="medium"),
            "Narration":    st.column_config.TextColumn("Narration",    width="large"),
            "Start Frame":  st.column_config.NumberColumn("Start Frame", format="%d"),
            "Stop Frame":   st.column_config.NumberColumn("Stop Frame",  format="%d"),
        },
    )

    st.markdown(
        "<p style='color:#555; font-size:0.8rem; margin-top:1rem;'>"
        "Results ranked by cosine similarity between query and video embeddings. Spit: test seen</p>",
        unsafe_allow_html=True,
    )

if st.session_state.page == "results" and st.session_state.results is None:
    st.session_state.page = "search"

if st.session_state.page == "search":
    page_search()
else:
    page_results()