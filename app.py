import shutil
from pathlib import Path

import streamlit as st

from forgeai.agent import Agent, load_definition
from forgeai.factory import Factory
from forgeai.llm import GeminiLLM

st.set_page_config(page_title="ForgeAI", layout="wide")
st.title("ForgeAI - the agent that builds agents")
tab_build, tab_chat = st.tabs(["1. Build an agent", "2. Chat with an agent"])

with tab_build:
    brief = st.text_area("One-paragraph brief", height=120)
    files = st.file_uploader("Documents (.txt .md .pdf .csv)", accept_multiple_files=True)
    name = st.text_input("Run name", "my_run")
    if st.button("Build agent", type="primary") and brief and files:
        docs = Path("runs") / name / "docs"
        shutil.rmtree(docs, ignore_errors=True)
        docs.mkdir(parents=True)
        for f in files:
            (docs / f.name).write_bytes(f.getvalue())
        box = st.empty()
        lines = []

        def log(msg):
            lines.append(str(msg))
            box.code("\n".join(lines))

        with st.spinner("Building... this takes a few minutes"):
            Factory(GeminiLLM(), brief, str(docs), f"runs/{name}/out", auto=True, log=log).run()
        st.success("Done")
        st.markdown(Path(f"runs/{name}/out/report.md").read_text(encoding="utf-8"))

with tab_chat:
    defs = sorted(str(p) for p in Path("runs").glob("*/agent_definition.json")) + \
           sorted(str(p) for p in Path("runs").glob("*/out/agent_definition.json"))
    if not defs:
        st.info("No agents yet.")
    else:
        choice = st.selectbox("Agent", defs)
        if st.session_state.get("loaded") != choice:
            d = load_definition(choice)
            st.session_state.agent = Agent(d, GeminiLLM(d.get("model")))
            st.session_state.history = []
            st.session_state.loaded = choice
        st.caption(f"tools: {st.session_state.agent.d['tools'] or 'none'}")
        for role, text in st.session_state.history:
            st.chat_message(role).write(text)
        q = st.chat_input("Ask the agent")
        if q:
            st.chat_message("user").write(q)
            r = st.session_state.agent.answer(q, st.session_state.history)
            st.chat_message("assistant").write(r["answer"])
            st.caption(f"sources: {r['sources']}  tool: {r['tool_used']}")
            st.session_state.history += [("user", q), ("assistant", r["answer"])]