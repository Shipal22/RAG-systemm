Features
PDF indexing with OCR fallback (PyMuPDF + Tesseract)
Document library: switch and delete indexed PDFs
Follow-up question rewriting, MMR retrieval, streaming answers
Source pages and text shown for every answer
Setup
bash
pip install streamlit pymupdf pytesseract pillow requests \
    langchain-huggingface langchain-chroma langchain-text-splitters \
    langchain-core sentence-transformers

Install Tesseract OCR, then add your key to .streamlit/secrets.toml:

toml
OPENROUTER_API_KEY = "sk-or-..."
Run
bash
streamlit run app.py

Upload a PDF, click Index document, then chat.
