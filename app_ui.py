import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="Arabic QA Interface", page_icon="?", layout="centered")
st.title("Arabic QA - One-Turn Interface")
st.caption("Ask one Arabic question, get category classification, Arabic answer, and English translations.")


@st.cache_data(ttl=15)
def fetch_models() -> list:
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Could not load models from API: {exc}")
        return []


def main() -> None:
    st.sidebar.header("Backend")
    api_url = st.sidebar.text_input("API base URL", value=API_BASE_URL)

    if st.sidebar.button("Refresh model list"):
        st.cache_data.clear()

    models = fetch_models() if api_url == API_BASE_URL else []
    if api_url != API_BASE_URL:
        try:
            response = requests.get(f"{api_url}/models", timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            models = response.json()
        except Exception as exc:
            st.error(f"Could not load models from custom URL: {exc}")
            models = []

    if not models:
        st.warning("No models available. Ensure FastAPI server is running and loaded correctly.")
        return

    model_options = [m["model_id"] for m in models]
    model_meta = {m["model_id"]: m for m in models}

    default_idx = 0
    for i, mid in enumerate(model_options):
        if model_meta[mid].get("ready", False):
            default_idx = i
            break

    selected_model = st.selectbox("Question-Answering Model", options=model_options, index=default_idx)
    selected_info = model_meta[selected_model]

    st.info(
        f"Family: {selected_info.get('family')} | "
        f"Preprocessing: {selected_info.get('preprocessing')} | "
        f"Embedding: {selected_info.get('embedding_strategy')} | "
        f"Ready: {selected_info.get('ready')}"
    )

    question = st.text_area("Question", height=120, placeholder="اكتب سؤالك هنا...")
    ask_btn = st.button("Generate Answer", type="primary")

    if ask_btn:
        if not question.strip():
            st.error("Please enter a question.")
            return

        payload = {"question": question, "model_id": selected_model}

        with st.spinner("Generating..."):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code >= 400:
                    st.error(f"Request failed ({response.status_code}): {response.text}")
                    return
                result = response.json()
            except Exception as exc:
                st.error(f"Error while requesting prediction: {exc}")
                return

        st.subheader("Result")
        st.write("Selected model:", result.get("selected_model"))

        cls = result.get("classification_result", {})
        st.markdown("### Classification")
        st.write("Label:", cls.get("label"))
        st.write("Confidence:", cls.get("confidence"))
        st.write("Model source:", cls.get("source"))

        st.markdown("### Arabic Answer")
        st.write(result.get("generated_answer_ar", ""))

        st.markdown("### English Translation")
        st.write("Question:", result.get("translated_question"))
        st.write("Answer:", result.get("translated_answer"))
        st.caption(f"Translation status: {result.get('translation_status', 'pending')}")

        with st.expander("Debug"):
            st.json(result.get("debug", {}))


if __name__ == "__main__":
    main()
