import os

import streamlit as st

st.set_page_config(page_title="Investment Analysis Hub", layout="wide")

# Title and Description
st.title("📊 Investment Analysis Hub")
st.subheader("Unlock Insights. Make Smarter Investment Decisions.")

st.markdown("""
Welcome to the **Investment Analysis Hub** – your central place to explore investment data, monitor market trends, and gain financial insights.
""")

st.markdown("---")

# Dynamic Page Detection
st.markdown("## 📄 Available Sections (visible in sidebar):")

PAGES_DIR = "pages"

if os.path.exists(PAGES_DIR):
    page_files = sorted(
        [
            f
            for f in os.listdir(PAGES_DIR)
            if f.endswith(".py") and not f.startswith("_")
        ]
    )

    for page in page_files:
        name = page.split(".")[0]
        # Remove leading numbers and underscores, replace underscores with spaces
        display_name = name.lstrip("0123456789_").replace("_", " ")
        st.markdown(f"- 🟢 **{display_name}**")
else:
    st.warning("No pages found in the `pages/` directory.")

st.markdown("---")

st.success("👉 Use the **sidebar** to navigate to each section.")
