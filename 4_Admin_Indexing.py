import streamlit as st
import subprocess, sys

st.title("Admin: Build Indexes")
st.write("Run local indexers to build lexical and vector stores from items.csv")

if st.button("Build Lexical + Embeddings"):
    cmd = [sys.executable, "services/indexer/main.py", "--build-all"]
    st.code(" ".join(cmd))
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        st.success("Indexer Completed")
        st.text(out)
    except subprocess.CalledProcessError as e:
        st.error("Indexer failed")
        st.text(e.output.decode())
