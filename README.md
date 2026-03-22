# AuditSight Pro

A **Streamlit** app for **Turkish invoices and receipts**: hybrid OCR extracts **net amount, VAT, and payable total**; results are checked with **Net + VAT = Total**; **Google Gemini** suggests an expense category and a short auditor-style note. Session history and **CSV / JSON** audit export are included.

### Status & roadmap

This project is a **prototype** — a learning and demo build (e.g. coursework / Future Talent Program), **not** certified accounting or audit software. OCR and model outputs can be wrong; always **verify amounts against the original document** before any real-world use.

The author **plans to keep improving** the app over time (accuracy, features, tests, deployment). Treat the current version as a starting point, not a final product.

---

## Features

| Area | Description |
|------|-------------|
| **Hybrid OCR** | Tesseract → EasyOCR → optional Gemini Vision on weak scans |
| **Field extraction** | Label-based parsing for payable total, VAT base, calculated VAT |
| **Compliance** | Tolerant validation of Net + VAT vs. payable total |
| **AI** | Gemini: category + two-sentence Turkish auditor note |
| **Session & export** | Sidebar history; downloadable audit trail |
| **UI** | Dark theme, glass-style layout |

---

## Tech stack

- Python 3.12 (recommended)
- Streamlit · Pandas · Pillow · OpenCV
- Tesseract OCR · EasyOCR
- `google-generativeai` (Gemini 1.5 Flash)

---

## Local setup

```bash
git clone <your-repo-url>
cd <project-folder>
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Install **Tesseract** on your system. If needed, set `TESSERACT_CMD` to the executable path.

### Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Recommended | Category, auditor note, optional Vision OCR |
| `AUDITSIGHT_LINKEDIN` / `AUDITSIGHT_GITHUB` | No | Footer links (defaults in `app.py`) |

For local Streamlit secrets, create `.streamlit/secrets.toml` — **do not commit it**:

```toml
GEMINI_API_KEY = "your-key"
```

---

## Streamlit Community Cloud

1. Connect the GitHub repo; main file: **`app.py`**
2. **Settings → Secrets:** set `GEMINI_API_KEY`
3. `packages.txt` / `runtime.txt` support system Python and apt packages

First EasyOCR run may download models (can take a while).

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

---

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI, session state, exports |
| `utils.py` | OCR, parsing, validation, Gemini enrichment |
| `tests/` | Pytest for core logic |

---

## Public vs private GitHub

- **Evaluation / portfolio / hiring:** Often reviewers expect a **public** repo or read access so they can browse code and history.
- **Private** is fine if the org only asks for a zip or invited access — follow their instructions.
- **Never** commit API keys, real `secrets.toml`, or personal sample data (e.g. `invoice_db.json` with sensitive content).

---

## Author

**Fatmanur Yılmaz** — YGA & UP School, Future Talent Program.
