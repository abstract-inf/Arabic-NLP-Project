## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend API (Terminal 1):

```bash
uvicorn app_api:app --reload
```

This launches the model API server.

Start the UI (Terminal 2):

```bash
streamlit run app_ui.py
```

This opens the Streamlit interface.
