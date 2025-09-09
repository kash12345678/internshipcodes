import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001/ask"

st.set_page_config(page_title="Community Assistance RAG", page_icon="🤝", layout="centered")

st.title("🤝 Community Assistance Information")
st.write("Ask about **Education, Food, Transport, Healthcare** and choose answer length.")

# Sidebar options
st.sidebar.header("Options")
topic = st.sidebar.selectbox(
    "Choose a topic:",
    ["education", "food", "transport", "healthcare"],
    key="topic_select"
)
detail = st.sidebar.radio("Answer type:", ["short", "long"], key="detail_radio")

# Add query input
query = st.text_input("Ask a specific question (optional):", key="query_input")

# Fixed button with unique key
if st.button("Get Answer", key="get_answer_button"):
    try:
        params = {"topic": topic, "detail": detail}
        if query:
            params["query"] = query
            
        response = requests.get(API_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            st.subheader(f"❓ Question: {data['question'].capitalize()}")
            st.write(f"**Answer:** {data['answer']}")

            st.markdown("### 🔗 References")
            for ref in data["references"]:
                st.markdown(f"- [{ref}]({ref})")

            st.markdown("### 🤔 Related Questions")
            for rq in data["related_questions"]:
                st.markdown(f"- {rq}")

        else:
            st.error(f"Error {response.status_code}: Could not fetch data")

    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        