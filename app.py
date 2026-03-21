# Audit-Sight/app.py
"""
AuditSight Pro — Streamlit entry (bulletproof data + premium dark UI).
"""
import uuid

import pandas as pd
import streamlit as st
from PIL import Image

from utils import (
    extract_invoice_fields,
    get_sustainability_stats,
    increment_sustainability_counter,
    init_sustainability_counter,
    parse_tr_number,
    run_ocr,
    suggest_accounting_code,
    validate_vat,
)

# MUST be first Streamlit command
st.set_page_config(
    page_title="AuditSight Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    .stApp {
        font-family: "Inter", sans-serif;
        color-scheme: dark;
        background: radial-gradient(1200px 600px at 10% 0%, rgba(88,101,242,0.22), rgba(14,17,23,0) 60%),
                    radial-gradient(900px 500px at 90% 20%, rgba(16,185,129,0.18), rgba(14,17,23,0) 55%),
                    linear-gradient(180deg, #0a0d12 0%, #06080c 100%) !important;
        background-attachment: fixed !important;
    }

    div.block-container {
        max-width: 1240px !important;
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        margin: auto !important;
    }

    section[data-testid="stSidebar"] {
        background: #1e1e26 !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    .s-card {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .footer-link:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,212,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_session_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("selected_record_id", None)


def _get_record_by_id(record_id):
    if not record_id:
        return None
    history = st.session_state.get("history")
    if not isinstance(history, list):
        return None
    for r in history:
        if isinstance(r, dict) and r.get("id") == record_id:
            return r
    return None


def _normalize_summary_to_fields(summary) -> dict:
    """
    Invoice fields live directly on current['summary'] as a flat dict.
    Legacy: summary may wrap fields under 'fields' — use only if flat keys are absent.
    """
    if summary is None or not isinstance(summary, dict):
        return {}
    nested = summary.get("fields")
    if isinstance(nested, dict) and not any(
        k in summary for k in ("firma_adi", "net_tutar", "kdv_tutari", "genel_toplam", "tarih")
    ):
        return dict(nested)
    out = {k: v for k, v in summary.items() if k != "fields"}
    return out


def _safe_money_cell(val) -> str:
    if val is None:
        return "0.00"
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        n = parse_tr_number(str(val))
        return f"{float(n):.2f}" if n is not None else "0.00"


def _fields_editor_df(fields: dict) -> pd.DataFrame:
    if not isinstance(fields, dict):
        fields = {}
    rows = [
        {"field": "Firma Adı", "value": str(fields.get("firma_adi") or "Not Found")},
        {"field": "Tarih", "value": str(fields.get("tarih") or "Not Found")},
        {"field": "Net Tutar", "value": _safe_money_cell(fields.get("net_tutar"))},
        {"field": "KDV Tutarı", "value": _safe_money_cell(fields.get("kdv_tutari"))},
        {"field": "Genel Toplam", "value": _safe_money_cell(fields.get("genel_toplam"))},
    ]
    return pd.DataFrame(rows)


def _df_to_fields(df) -> dict:
    if df is None or not isinstance(df, pd.DataFrame):
        return {}
    try:
        rows_iter = df.iterrows()
    except Exception:
        return {}
    out_map = {}
    for _, row in rows_iter:
        try:
            k = row.get("field") if hasattr(row, "get") else row["field"]
            v = row.get("value") if hasattr(row, "get") else row["value"]
        except (TypeError, KeyError, AttributeError):
            continue
        if k is not None:
            out_map[k] = v
    return {
        "firma_adi": out_map.get("Firma Adı", "Not Found"),
        "tarih": out_map.get("Tarih", "Not Found"),
        "net_tutar": parse_tr_number(str(out_map.get("Net Tutar", "0"))),
        "kdv_tutari": parse_tr_number(str(out_map.get("KDV Tutarı", "0"))),
        "genel_toplam": parse_tr_number(str(out_map.get("Genel Toplam", "0"))),
    }


_init_session_state()
init_sustainability_counter()

# --- Sidebar
with st.sidebar:
    st.markdown(
        "<h2 style='color:white;margin-bottom:0;'>AuditSight Pro</h2>",
        unsafe_allow_html=True,
    )
    st.divider()
    
    
    st.info("""
    **🔍 Sistem Özeti**
    AI destekli finansal denetim aracı.  
    **Net + KDV = Toplam** algoritmasıyla veri doğruluğunu garanti eder.
    """)
    
    tab_dash, tab_hist = st.tabs(["DASHBOARD", "GEÇMİŞ"])

    with tab_dash:
        stats = get_sustainability_stats()
        if not isinstance(stats, dict):
            stats = {}
        inv = float(stats.get("invoices", 0.0) or 0.0)
        paper = float(stats.get("paper_saved_g", 0.0) or 0.0)
        trees = (paper / 1000.0) / 80.0
        st.markdown(
            f"""
            <div class="s-card"><div class="s-label">📄 İŞLENEN FATURA</div>
            <div class="s-value">{inv:.0f}</div></div>
            <div class="s-card"><div class="s-label">♻️ KAĞIT TASARRUFU</div>
            <div class="s-value">{paper:.2f}g</div></div>
            <div class="s-card"><div class="s-label">🌲 AĞAÇ EŞDEĞERİ</div>
            <div class="s-value">{trees:.3f} Adet</div></div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(inv / 100.0, 0.0), 1.0))

    with tab_hist:
        hist = st.session_state.get("history")
        if isinstance(hist, list) and hist:
            labels = [
                f"{(r or {}).get('filename', 'Unknown')} — {(r or {}).get('id', 'NA')}"
                for r in reversed(hist)
                if isinstance(r, dict)
            ]
            if labels:
                picked = st.selectbox("Geçmiş Faturalar", options=labels)
                if picked and " — " in picked:
                    st.session_state.update(
                        {"selected_record_id": picked.rsplit(" — ", 1)[-1].strip()}
                    )
            else:
                st.info("Henüz kayıt yok.")
        else:
            st.info("Henüz geçmiş bulunmuyor.")

st.markdown(
    '<div class="brandbar"><div class="brand-title">Smart Tax & Invoice Auditor</div></div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([0.4, 0.6], gap="large")

# Streamlit tüm script'i yeniden çalıştırır; NameError önlemi
uploaded = None
do_analyze = False

with col_left:
    st.subheader("📄 Fatura Yükleme")
    with st.form("upload_form", clear_on_submit=False):
        uploaded = st.file_uploader(
            "Dosya seçin",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        do_analyze = st.form_submit_button(
            "🔍 Analiz Et", type="primary", use_container_width=True
        )
    if uploaded:
        st.image(Image.open(uploaded), use_container_width=True, caption="Yüklenen Fatura")

if do_analyze and uploaded:
    try:
        with st.spinner("OCR analizi yapılıyor…"):
            img = Image.open(uploaded)
            ocr = run_ocr(img)
            if not isinstance(ocr, dict):
                ocr = {}
            ocr_lines = ocr.get("lines")
            fields_new = extract_invoice_fields(
                str(ocr.get("text") or ""),
                ocr_lines if isinstance(ocr_lines, list) else None,
            )
            if not isinstance(fields_new, dict):
                fields_new = {}
            rec_id = str(uuid.uuid4())[:8]
            record = {
                "id": rec_id,
                "filename": getattr(uploaded, "name", None) or "upload",
                "ocr_text": str(ocr.get("text") or ""),
                "ocr_lines": ocr.get("lines") if isinstance(ocr.get("lines"), list) else [],
                "summary": fields_new,
            }
            hist = st.session_state.get("history")
            if not isinstance(hist, list):
                hist = []
            hist.append(record)
            st.session_state.update({"history": hist, "selected_record_id": rec_id})
            increment_sustainability_counter(1)
            st.rerun()
    except Exception as exc:
        st.session_state["ocr_last_error"] = str(exc)
        st.warning(
            "OCR sırasında sorun oluştu; tekrar deneyin veya farklı bir görüntü kullanın."
        )

# --- Right panel
current = _get_record_by_id(st.session_state.get("selected_record_id"))
if current is None or not isinstance(current, dict):
    current = {}

with col_right:
    if current:
        if not isinstance(fields, dict):
            fields = {}

        edit_col, info_col = st.columns([1, 1], gap="medium")
        with edit_col:
            st.markdown("**Veri Düzenleme**")
            edited = st.data_editor(
                _fields_editor_df(fields),
                use_container_width=True,
                hide_index=True,
            )
            e_fields = _df_to_fields(edited)
        if e_fields is None or not isinstance(e_fields, dict):
            e_fields = {}

        safe = {
            "firma_adi": e_fields.get("firma_adi") or "Not Found",
            "tarih": e_fields.get("tarih") or "Not Found",
            "net_tutar": e_fields.get("net_tutar") if e_fields.get("net_tutar") is not None else 0.0,
            "kdv_tutari": e_fields.get("kdv_tutari") if e_fields.get("kdv_tutari") is not None else 0.0,
            "genel_toplam": e_fields.get("genel_toplam") if e_fields.get("genel_toplam") is not None else 0.0,
        }

        with info_col:
            st.markdown("**Denetim Raporu**")
            try:
                override_applied = bool(fields.get("total_override_applied", False))
                total_source = str(fields.get("total_source") or "")
                if override_applied:
                    st.info(
                        f"🧮 Genel Toplam matematik doğrulaması ile düzeltildi: {_safe_money_cell(safe.get('genel_toplam'))} (kaynak: {total_source or 'calculated_override'})"
                    )
                val = validate_vat(
                    safe.get("net_tutar"),
                    safe.get("kdv_tutari"),
                    safe.get("genel_toplam"),
                )
                if not isinstance(val, dict):
                    val = {}
                c = val.get("color", "gray")
                lbl = val.get("label", "Durum bilinmiyor.")
                if c == "green":
                    st.success(str(lbl))
                elif c == "orange":
                    st.warning(str(lbl))
                elif c == "red":
                    # Kırmızı st.error kutusu yerine uyarı (denetim riski ≠ uygulama hatası)
                    st.warning(str(lbl))
                else:
                    st.info(str(lbl))

                if e_fields is None:
                    e_fields = {}
                if current is None:
                    current = {}
                if not isinstance(e_fields, dict):
                    e_fields = {}
                if not isinstance(current, dict):
                    current = {}
                code = suggest_accounting_code(
                    e_fields.get("firma_adi", "Bilinmiyor"),
                    str(current.get("ocr_text") or ""),
                )
                st.info(f"🏷️ **Önerilen Hesap Kodu:** {code or 'Belirlenemedi'}")
            except Exception:
                st.info("Denetim özeti gösterilemedi; tablodaki değerleri kontrol edin.")
                st.info("🏷️ **Önerilen Hesap Kodu:** Belirlenemedi")
    else:
        st.info("💡 Lütfen bir fatura yükleyip analiz edin.")

# --- Footer
st.markdown(
    """
    <style>
    .footer-container {
        text-align: center;
        padding: 40px 20px;
        margin-top: 60px;
        border-top: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        border-radius: 20px;
    }
    .footer-title {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 20px;
    }
    .author-name { font-size: 1.1em; color: #eee; margin-bottom: 25px; }
    .author-name span { color: #00d4ff; font-weight: 800; }
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin-bottom: 25px;
        flex-wrap: wrap;
    }
    .footer-link {
        text-decoration: none !important;
        color: #bbb !important;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 22px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 30px;
        transition: all 0.3s ease;
        font-weight: 600;
        font-size: 14px;
    }
    .footer-link:hover {
        color: #fff !important;
        border-color: #00d4ff;
        background: rgba(0,212,255,0.1);
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,212,255,0.2);
    }
    .footer-icon { width: 18px; height: 18px; filter: brightness(0) invert(1); }
    .footer-note { font-size: 12px; color: rgba(255,255,255,0.3); margin-top: 15px; }
    </style>

    <div class="footer-container">
        <div class="footer-title">İletişim &amp; Kaynak Kodlar</div>
        <div class="author-name">Concept, Design &amp; Development by <span>Fatmanur Yılmaz</span></div>
        <div class="footer-links">
            <a class="footer-link" href="https://github.com/yilmazfatmanur" target="_blank" rel="noopener noreferrer">
                <img class="footer-icon" src="https://cdn.simpleicons.org/github/ffffff" alt="GitHub">
                GitHub
            </a>
            <a class="footer-link" href="https://www.linkedin.com/in/fatmanur-y%C4%B1lmaz" target="_blank" rel="noopener noreferrer">
                <img class="footer-icon" src="https://cdn.simpleicons.org/linkedin/ffffff" alt="LinkedIn">
                LinkedIn
            </a>
        </div>
        <div class="footer-note">© 2026 | AuditSight Pro | Future Talent Program </div>
    </div>
    """,
    unsafe_allow_html=True,
)
