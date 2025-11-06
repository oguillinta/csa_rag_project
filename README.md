# Customer Support Agent RAG Chatbot (Beginner-Friendly)

This small project shows how to load CSV rows into a vector database (Chroma),
use OpenAI to generate embeddings and answers, and chat in a *Customer Support Agent* role.
The code is intentionally simple so it’s easier to read and learn from.

## Project layout
```
csa_rag_project/
  data/
    orders.csv
  src/
    app.py            # Streamlit app (UI)
    data_loader.py    # Reads CSV files and converts rows to text
    vector_store.py   # Chroma (add/search)
    rag_service.py    # Retrieval + OpenAI calls
    config.py         # Small settings and helpers
  requirements.txt
  README.md
```

## Quick start
1. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key (or paste it inside the app sidebar):
   ```bash
   export OPENAI_API_KEY=sk-...
   # Windows PowerShell:
   # $env:OPENAI_API_KEY="sk-..."
   ```

4. Run the app:
   ```bash
   streamlit run src/app.py
   ```

5. In the app:
   - Click **Rebuild Index** to index the sample `orders.csv` (or upload your own CSV).
   - Ask questions like: *"What's the status of order 1001?"*

## Notes
- The bot only answers if info is present in the CSV and if the question is in-scope for the **Customer Support Agent** role (order status, cancellations, returns, shipping dates, delivery issues).
- If it can't find good matches or the question is out of scope, it will reply with a short, polite refusal.
