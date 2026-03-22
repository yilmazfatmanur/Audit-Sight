# Audit-Sight/app.py
"""AuditSight Pro — High-end Fintech terminal UI (dark glass + hybrid OCR)."""
from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# set_page_config, utils importundan ÖNCE olmalı (utils içinde @st.cache_resource kaydı var).
st.set_page_config(
    page_title="AuditSight Pro — Fatura denetimi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from PIL import Image

from utils import (
    extract_invoice_fields,
    gemini_invoice_enrichment,
    gemini_is_configured,
    get_paper_saved_grams,
    increment_sustainability_counter,
    infer_invoice_category_heuristic,
    init_sustainability_counter,
    parse_tr_number,
    run_ocr,
    validate_golden_rule,
)


def _api_key() -> str:
    try:
        s = st.secrets.get("GEMINI_API_KEY", "")
        if s:
            return str(s).strip()
    except Exception:
        pass
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


# Kartvizit — secrets/env ile üzerine yazılabilir
_DEFAULT_LINKEDIN = "https://www.linkedin.com/in/fatmanur-yılmaz"
_DEFAULT_GITHUB = "https://github.com/yilmazfatmanur"


def _ensure_https_url(url: str) -> str:
    """Secrets/env'de 'www....' gibi şemasız adresler için https ekler."""
    u = (url or "").strip()
    if not u:
        return u
    if u.startswith("//"):
        return "https:" + u
    if not (u.startswith("http://") or u.startswith("https://")):
        return "https://" + u
    return u


def _social_urls() -> tuple[str, str]:
    li, gh = "", ""
    try:
        sec = st.secrets
        li = str(sec.get("LINKEDIN_URL", "") or "").strip()
        gh = str(sec.get("GITHUB_URL", "") or "").strip()
    except Exception:
        pass
    if not li:
        li = (os.environ.get("AUDITSIGHT_LINKEDIN") or _DEFAULT_LINKEDIN).strip()
    if not gh:
        gh = (os.environ.get("AUDITSIGHT_GITHUB") or _DEFAULT_GITHUB).strip()
    return _ensure_https_url(li), _ensure_https_url(gh)


def _metric_value(v) -> str:
    if v is None:
        return "Analiz ediliyor…"
    try:
        return f"{float(v):,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        n = parse_tr_number(str(v))
        if n is None:
            return "Analiz ediliyor…"
        return f"{n:,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")


def _init_session_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("selected_record_id", None)
    st.session_state.setdefault("last_analyzed_upload_sig", None)
    # Son başarılı analiz kaydı (sidebar ile yeni dosya beklemesi çakışmasını çözmek için)
    st.session_state.setdefault("last_completed_record_id", None)


def _upload_sig(uploaded_file) -> tuple[str, int] | None:
    """Yüklenen dosyayı benzersizleştir (ad + boyut) — önizleme değişince sağ paneli senkronlamak için."""
    if uploaded_file is None:
        return None
    return (
        str(getattr(uploaded_file, "name", "") or ""),
        int(getattr(uploaded_file, "size", None) or 0),
    )


def _get_record_by_id(record_id):
    if not record_id:
        return None
    hist = st.session_state.get("history")
    if not isinstance(hist, list):
        return None
    for r in hist:
        if isinstance(r, dict) and r.get("id") == record_id:
            return r
    return None


def _trees_saved_from_invoices(invoice_count: int) -> float:
    """Eco-Impact: işlenen fatura sayısı × 0.00002 (eşdeğer ağaç)."""
    return max(0, int(invoice_count)) * 0.00002


def _format_total_compact(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.2f} TL".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "—"


def _export_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _audit_rows_for_export(valid_hist: list) -> list[dict]:
    """Oturum geçmişi — tablo + doğrulama durumu (CSV için düz satırlar)."""
    rows: list[dict] = []
    for r in valid_hist:
        if not isinstance(r, dict):
            continue
        summ = r.get("summary") if isinstance(r.get("summary"), dict) else {}
        net = summ.get("net_tutar")
        kdv = summ.get("kdv_tutari")
        top = summ.get("genel_toplam")
        vr = validate_golden_rule(net, kdv, top)
        ok = vr.get("ok")
        if ok is True:
            durum = "GÜVENLİ"
        elif ok is False:
            durum = "RİSKLİ"
        else:
            durum = "EKSİK_VERİ"
        note = str(r.get("ai_note") or "").replace("\r\n", " ").replace("\n", " ")
        rows.append(
            {
                "Dosya": r.get("filename"),
                "Kimlik": r.get("id"),
                "Firma": summ.get("firma_adi"),
                "Kategori": r.get("ai_category_tr"),
                "Net (TL)": net,
                "KDV (TL)": kdv,
                "Toplam (TL)": top,
                "Doğrulama": durum,
                "Doğrulama mesajı": (vr.get("message") or vr.get("detail") or "")[:500],
                "Denetçi notu": note[:4000],
            }
        )
    return rows


def _audit_json_payload(valid_hist: list) -> dict:
    """Tam denetim izi: kayıt başına özet + altın kural + OCR önizleme."""
    recs: list[dict] = []
    for r in valid_hist:
        if not isinstance(r, dict):
            continue
        summ = r.get("summary") if isinstance(r.get("summary"), dict) else {}
        net = summ.get("net_tutar")
        kdv = summ.get("kdv_tutari")
        top = summ.get("genel_toplam")
        vr = validate_golden_rule(net, kdv, top)
        ocr = str(r.get("ocr_text") or "")
        recs.append(
            {
                "id": r.get("id"),
                "filename": r.get("filename"),
                "summary": summ,
                "ai_category_tr": r.get("ai_category_tr"),
                "ai_category_en": r.get("ai_category_en"),
                "ai_enrich_source": r.get("ai_enrich_source"),
                "ai_note": r.get("ai_note"),
                "golden_rule": {
                    "ok": vr.get("ok"),
                    "label": vr.get("label"),
                    "detail": vr.get("detail"),
                    "message": vr.get("message"),
                },
                "ocr_text_preview": ocr[:1200],
            }
        )
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "app": "AuditSight Pro",
        "export_version": 1,
        "record_count": len(recs),
        "records": recs,
    }


st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
    .stApp {
        font-family: 'Inter', 'Roboto', 'Segoe UI', system-ui, sans-serif;
        background: radial-gradient(ellipse 120% 80% at 50% -20%, #0f172a 0%, #0a0e14 45%, #06080c 100%);
        color: #e2e8f0;
    }
    div.block-container { max-width: 1400px; padding-top: 0.75rem; padding-bottom: 0.5rem; }
    section[data-testid="stMain"] { color: #e2e8f0; }
    section[data-testid="stMain"] .stMarkdown, section[data-testid="stMain"] p, section[data-testid="stMain"] label { color: #cbd5e1; }
    /* Neon glass — kartlar */
    section[data-testid="stMain"] form[data-testid="stForm"],
    section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(0, 255, 204, 0.28) !important;
        border-radius: 16px !important;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        padding: 1rem 1.1rem 1.15rem 1.1rem !important;
    }
    section[data-testid="stMain"] [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 255, 204, 0.22);
        border-radius: 16px;
    }
    section[data-testid="stMain"] [data-testid="stMetricContainer"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 255, 204, 0.25) !important;
        border-radius: 16px !important;
        box-shadow: 0 0 16px rgba(0, 255, 204, 0.06);
        padding: 0.5rem 0.4rem 0.6rem 0.4rem !important;
    }
    section[data-testid="stMain"] [data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        color: #94a3b8 !important;
    }
    section[data-testid="stMain"] [data-testid="stMetricValue"] {
        font-variant-numeric: tabular-nums;
        color: #f1f5f9 !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }
    /* DataFrame / tablo */
    section[data-testid="stMain"] [data-testid="stDataFrame"] {
        border: 1px solid rgba(0, 255, 204, 0.22);
        border-radius: 16px;
        overflow: hidden;
    }
    section[data-testid="stMain"] div[data-testid="stDataFrame"] > div {
        background: rgba(255, 255, 255, 0.02) !important;
    }
    /* Genel linkler */
    section[data-testid="stMain"] a[href^="http"] {
        transition: color 0.3s ease, transform 0.3s ease, filter 0.3s ease;
    }
    section[data-testid="stMain"] a[href*="linkedin"]:hover {
        color: #00a0dc !important;
        transform: translateY(-2px) scale(1.03);
        filter: drop-shadow(0 0 8px rgba(0, 160, 220, 0.45));
    }
    section[data-testid="stMain"] a[href*="github"]:hover {
        color: #fafafa !important;
        transform: translateY(-2px) scale(1.03);
        filter: drop-shadow(0 0 6px rgba(250, 250, 250, 0.25));
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #0a0d12 100%);
        border-right: 1px solid rgba(0, 255, 204, 0.12);
    }
    /* Zero Waste — yeşil vurgu (#00cc88) */
    section[data-testid="stSidebar"] [data-testid="stMetricContainer"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(0, 204, 136, 0.35) !important;
        border-radius: 16px !important;
        padding: 0.75rem 0.55rem !important;
        box-shadow: 0 0 18px rgba(0, 204, 136, 0.12);
    }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
        color: #5eead4 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #00cc88 !important;
    }
    .hero-gradient {
        font-size: clamp(2rem, 4.2vw, 2.85rem);
        font-weight: 800;
        letter-spacing: -0.035em;
        line-height: 1.12;
        margin: 0;
        background: linear-gradient(100deg, #00ffcc 0%, #22d3ee 35%, #818cf8 70%, #c084fc 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }
    .hero-brand-line { margin: 0 0 2rem 0; line-height: 1.2; }
    .hero-brand-line .muted {
        font-size: clamp(0.95rem, 2vw, 1.1rem);
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #94a3b8;
        -webkit-text-fill-color: #94a3b8;
    }
    .panel-label {
        font-size: 1.02rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #94a3b8;
        margin: 0 0 1.35rem 0;
        padding-top: 0.5rem;
    }
    .zw-pill {
        display: inline-block;
        margin: 0 0 0.75rem 0;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.14em;
        color: #022c22;
        background: linear-gradient(90deg, #00cc88, #34d399);
        border: 1px solid rgba(0, 255, 204, 0.5);
    }
    .tech-stack-wrap { font-size: 0.88rem; color: #94a3b8; line-height: 1.85; margin: 4px 0 0 0; }
    .tech-stack-wrap .row { display: flex; align-items: center; gap: 0.65rem; min-height: 1.6rem; }
    .tech-stack-wrap .ico { width: 1.35rem; text-align: center; flex-shrink: 0; font-size: 1rem; }
    .tech-stack-wrap .txt { color: #e2e8f0; font-weight: 500; }
    .empty-audit-box {
        border: 1px dashed rgba(0, 255, 204, 0.25);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(14px);
        padding: 2.25rem 1.5rem;
        text-align: center;
        color: #94a3b8;
    }
    .empty-audit-box .t1 { font-size: 1.05rem; font-weight: 600; color: #e2e8f0; margin: 0 0 0.5rem 0; letter-spacing: 0.04em; }
    .empty-audit-box .t2 { font-size: 0.88rem; margin: 0; color: #64748b; line-height: 1.5; }
    .audit-card-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #64748b;
        margin: 0 0 0.75rem 0;
    }
    /* Kurumsal footer şeridi — koyu gri + neon çizgi */
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.footer-block) {
        background: linear-gradient(180deg, #1e232b 0%, #161a22 100%);
        border: 1px solid rgba(0, 255, 204, 0.22);
        border-radius: 16px;
        padding: 1.35rem 1rem 1.5rem 1rem;
        margin-top: 2.25rem;
        box-shadow: 0 -6px 40px rgba(0, 0, 0, 0.35), 0 0 24px rgba(0, 255, 204, 0.05);
    }
    .footer-block { color: #e2e8f0; }
    .footer-sub { color: #94a3b8; font-size: 0.88rem; margin-top: 0.35rem; }
    .footer-heading { color: #00ffcc; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; margin: 0 0 0.65rem 0; }
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.footer-block) a[data-testid="stLinkButton"] {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(.footer-block) a[data-testid="stLinkButton"]:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 14px rgba(0, 255, 204, 0.25);
    }
    section[data-testid="stSidebar"] h3 { font-size: 1.02rem !important; margin: 0.5rem 0 0.35rem 0 !important; color: #f1f5f9 !important; font-weight: 700 !important; }
    /* Mobile-style sidebar (AuditSight reference) */
    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0 0 1rem 0;
        padding: 0.25rem 0 0 0;
    }
    .sidebar-summary-box {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.22) 0%, rgba(30, 64, 175, 0.18) 100%);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 14px;
        padding: 14px 14px 16px 14px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .sidebar-summary-box .sum-icon { font-size: 1.25rem; margin-bottom: 0.5rem; }
    .sidebar-summary-box .sum-title {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #93c5fd;
        margin-bottom: 0.45rem;
    }
    .sidebar-summary-box .sum-body {
        font-size: 0.88rem;
        line-height: 1.55;
        color: #e2e8f0;
        margin: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stTabs"] {
        margin-top: 0.25rem;
    }
    section[data-testid="stSidebar"] [data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.12em !important;
    }
    section[data-testid="stSidebar"] [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    section[data-testid="stSidebar"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #f87171 !important;
        border-bottom: 2px solid #ef4444 !important;
    }
    .sidebar-stat-grid {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 0.75rem 0 0.85rem 0;
    }
    .sidebar-stat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 12px 16px 14px;
    }
    .sidebar-stat-card .ssc-icon { font-size: 1.2rem; margin-bottom: 6px; opacity: 0.95; }
    .sidebar-stat-card .ssc-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: #94a3b8;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .sidebar-stat-card .ssc-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f1f5f9;
        font-variant-numeric: tabular-nums;
    }
    section[data-testid="stSidebar"] .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #22c55e) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_init_session_state()
init_sustainability_counter()
_li_url, _gh_url = _social_urls()

hist = st.session_state.get("history")
if not isinstance(hist, list):
    hist = []
valid_hist = [r for r in hist if isinstance(r, dict)]
# Eski oturumlar: son tamamlanan kayıt + seçili panel (sidebar ile tutarlı)
if valid_hist:
    _vids = {r.get("id") for r in valid_hist if r.get("id")}
    if st.session_state.get("last_completed_record_id") is None:
        st.session_state["last_completed_record_id"] = valid_hist[-1].get("id")
    if not st.session_state.get("selected_record_id"):
        st.session_state["selected_record_id"] = valid_hist[-1].get("id")
    elif st.session_state.get("selected_record_id") not in _vids:
        st.session_state["selected_record_id"] = valid_hist[-1].get("id")
_trees_eq = _trees_saved_from_invoices(len(valid_hist))

paper = get_paper_saved_grams()
_inv_count = len(valid_hist)

with st.sidebar:
    st.markdown(
        '<p class="sidebar-brand-title">AuditSight Pro</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sidebar-summary-box">
            <div class="sum-icon">🔍</div>
            <div class="sum-title">Sistem özeti</div>
            <p class="sum-body">
                <strong>AuditSight Pro</strong>, fiş ve fatura görsellerinden hibrit OCR ile net, KDV ve
                ödenecek tutarı çıkarır; <strong>Net + KDV = Toplam</strong> kuralıyla anında
                tutarlılık kontrolü yapar. Gemini ile gider kategorisi ve kısa denetçi özeti üretir;
                analizler bu oturumda geçmişe kaydedilir.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_dash, tab_hist = st.tabs(["PANO", "GEÇMİŞ"])

    with tab_dash:
        st.markdown(
            f"""
            <div class="sidebar-stat-grid">
                <div class="sidebar-stat-card">
                    <div class="ssc-icon">📄</div>
                    <div class="ssc-label">İşlenen fatura</div>
                    <div class="ssc-value">{_inv_count}</div>
                </div>
                <div class="sidebar-stat-card">
                    <div class="ssc-icon">♻️</div>
                    <div class="ssc-label">Kağıt tasarrufu</div>
                    <div class="ssc-value">{paper:.2f} g</div>
                </div>
                <div class="sidebar-stat-card">
                    <div class="ssc-icon">🌲</div>
                    <div class="ssc-label">Ağaç eşdeğeri</div>
                    <div class="ssc-value">{_trees_eq:.3f} adet</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(_inv_count / 100.0, 0.0), 1.0))

        with st.expander("Teknoloji yığını", expanded=False):
            st.markdown(
                """
                <div class="tech-stack-wrap">
                    <div class="row"><span class="ico">🐍</span><span class="txt">Python</span></div>
                    <div class="row"><span class="ico">🖼️</span><span class="txt">OpenCV</span></div>
                    <div class="row"><span class="ico">✨</span><span class="txt">Google Gemini 1.5 Flash</span></div>
                    <div class="row"><span class="ico">📊</span><span class="txt">Streamlit</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_hist:
        if valid_hist:
            labels = [
                f"{(r or {}).get('filename', '?')} — {(r or {}).get('id', '')}"
                for r in reversed(valid_hist)
            ]
            _sel_id = st.session_state.get("selected_record_id")
            _idx = 0
            if _sel_id and labels:
                for _i, _lab in enumerate(labels):
                    if _lab.endswith(f" — {_sel_id}"):
                        _idx = _i
                        break
            _idx = max(0, min(_idx, len(labels) - 1))
            picked = st.selectbox(
                "Panelde göster",
                options=labels,
                index=_idx,
            )
            if picked and " — " in picked:
                st.session_state["selected_record_id"] = picked.rsplit(" — ", 1)[-1].strip()
        else:
            st.info("Henüz kayıtlı denetim yok. Ana sayfada **Analiz et** ile başlayın.")

        st.divider()
        if valid_hist:
            st.caption("Son kayıtlar")
            recent = list(reversed(valid_hist[-3:]))
            for r in recent:
                summ = r.get("summary") if isinstance(r.get("summary"), dict) else {}
                firma = str(summ.get("firma_adi") or r.get("filename") or "—").strip()
                if len(firma) > 36:
                    firma = firma[:33] + "…"
                top_amt = summ.get("genel_toplam")
                st.markdown(
                    f"**{firma}**  \n"
                    f"<span style='color:#8b949e;font-size:0.85rem;'>Toplam: {_format_total_compact(top_amt)}</span>",
                    unsafe_allow_html=True,
                )
            if len(valid_hist) > 3:
                st.caption(f"+ {len(valid_hist) - 3} eski kayıt")

st.markdown(
    """
    <div class="hero-brand-line">
        <span class="hero-gradient">AuditSight Pro</span>
        <span class="muted"> | AuditSight Pro</span>
    </div>
    """,
    unsafe_allow_html=True,
)

main_l, main_r = st.columns([1.2, 1.0], gap="large")

uploaded = None
go = False

with main_l:
    st.markdown('<p class="panel-label">Kaynak belge</p>', unsafe_allow_html=True)
    # file_uploader formun DIŞINDA olmalı: form içindeki widget değerleri yalnızca submit ile
    # sunucuya gelir; yeni fiş seçildiğinde sağ panelin senkronu için anında güncelleme gerekir.
    uploaded = st.file_uploader(
        "Fatura yükle (PNG / JPG)",
        type=["png", "jpg", "jpeg"],
        key="audit_invoice_upload",
        label_visibility="visible",
    )
    with st.form("upload_form"):
        go = st.form_submit_button("Analiz et", type="primary", use_container_width=True)
    if uploaded:
        st.image(Image.open(uploaded), use_container_width=True, caption="Belge önizleme")

if go and uploaded:
    try:
        ak = _api_key()
        img = Image.open(uploaded)
        with st.spinner("Belge analiz ediliyor…"):
            ocr = run_ocr(img, gemini_api_key=ak or None)
            if not isinstance(ocr, dict):
                ocr = {}
            text = str(ocr.get("text") or "")
            lines = ocr.get("lines") if isinstance(ocr.get("lines"), list) else None
            fields = extract_invoice_fields(text, lines)
        with st.spinner("Yapay zekâ: kategori ve denetçi notu…"):
            enrich = gemini_invoice_enrichment(text, fields, ak or None)
        rid = str(uuid.uuid4())[:8]
        rec = {
            "id": rid,
            "filename": getattr(uploaded, "name", None) or "fatura",
            "summary": fields,
            "ai_note": str(enrich.get("auditor_note") or ""),
            "ai_category_tr": enrich.get("category_tr"),
            "ai_category_en": enrich.get("category_en"),
            "ai_enrich_source": enrich.get("source"),
            "ocr_text": text[:8000],
        }
        h = st.session_state.get("history")
        if not isinstance(h, list):
            h = []
        h.append(rec)
        st.session_state.update(
            {
                "history": h,
                "selected_record_id": rid,
                "last_analyzed_upload_sig": _upload_sig(uploaded),
                "last_completed_record_id": rid,
            }
        )
        increment_sustainability_counter(1)
        st.rerun()
    except Exception:
        st.warning("İşlem tamamlanamadı. Lütfen tekrar deneyin.")

cur = _get_record_by_id(st.session_state.get("selected_record_id"))
if not isinstance(cur, dict):
    cur = {}

# Yeni fiş seçildi ama analiz edilmedi — SADECE panel hâlâ "son tamamlanan" kaydı gösteriyorsa uyar.
# Geçmişten başka fatura seçildiğinde tutarlar o kayda göre güncellenir (aynı kalma hatası giderilir).
_upload_sig_now = _upload_sig(uploaded) if uploaded is not None else None
_last_sig = st.session_state.get("last_analyzed_upload_sig")
_last_done = st.session_state.get("last_completed_record_id")
_sel = st.session_state.get("selected_record_id")
upload_pending = (
    _upload_sig_now is not None
    and _upload_sig_now != _last_sig
    and _last_done is not None
    and _sel == _last_done
)

with main_r:
    st.markdown('<p class="panel-label">Denetim istihbaratı</p>', unsafe_allow_html=True)
    if upload_pending:
        st.info(
            "Yeni fiş seçildi. **Tutar çıkarımı** ve diğer sonuçlar bu görüntü için "
            "henüz üretilmedi — sol tarafta **Analiz et**'e basın."
        )
        st.markdown(
            """
            <div class="empty-audit-box">
                <p class="t1">Analiz bekleniyor</p>
                <p class="t2">Önizleme güncellendi; tutarları görmek için <strong>Analiz et</strong> ile işleyin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif cur:
        fields = cur.get("summary") if isinstance(cur.get("summary"), dict) else {}
        net = fields.get("net_tutar")
        kdv = fields.get("kdv_tutari")
        top = fields.get("genel_toplam")
        isk = fields.get("tahmini_iskonto")
        brut = fields.get("brut_mal_hizmet")

        st.markdown('<p class="audit-card-title">Tutar çıkarımı</p>', unsafe_allow_html=True)
        try:
            amt_card = st.container(border=True)
        except TypeError:
            amt_card = st.container()
        with amt_card:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Net (KDV matrahı)", _metric_value(net))
            with c2:
                st.metric("KDV", _metric_value(kdv))
            with c3:
                st.metric("Ödenecek toplam", _metric_value(top))
            if brut is not None or isk is not None:
                st.caption(
                    f"Brüt / iskonto ipuçları: {_metric_value(brut)} · {_metric_value(isk)}"
                )

        cat_tr = cur.get("ai_category_tr")
        cat_en = cur.get("ai_category_en")
        if not cat_tr and cur.get("ocr_text"):
            _hc = infer_invoice_category_heuristic(str(cur.get("ocr_text") or ""))
            cat_tr = _hc.get("category_tr")
            cat_en = _hc.get("category_en")
        if cat_tr:
            _line = f"**Fiş / fatura kategorisi:** {cat_tr}"
            if cat_en:
                _line += f" — *{cat_en}*"
            st.info(_line)

        vr = validate_golden_rule(net, kdv, top)
        st.markdown('<p class="audit-card-title">Uyumluluk kontrolü</p>', unsafe_allow_html=True)
        try:
            val_card = st.container(border=True)
        except TypeError:
            val_card = st.container()
        with val_card:
            ok_msg = vr.get("message") or f"{vr.get('label', '')}: {vr.get('detail', '')}".strip(": ")
            if vr.get("box") == "green" and vr.get("ok") is True:
                st.success(ok_msg)
            elif vr.get("box") == "red" and vr.get("ok") is False:
                st.error(ok_msg)
            else:
                st.info(ok_msg)

        note = str(cur.get("ai_note") or "").strip()
        ai_enabled = gemini_is_configured(_api_key())
        if ai_enabled or note:
            st.markdown('<p class="audit-card-title">Yapay zekâ denetçi özeti</p>', unsafe_allow_html=True)
            try:
                analyst_box = st.container(border=True)
            except TypeError:
                analyst_box = st.container()
            with analyst_box:
                if note:
                    st.write(note)
                elif ai_enabled:
                    st.caption(
                        "Metin dönmedi (OCR çok kısa veya API yanıtı boş). "
                        "Daha net bir tarama ile yeniden deneyin."
                    )

        with st.expander("Çıkarılan tutarları düzenle"):
            n_in = st.text_input("Net", value="" if net is None else str(net))
            k_in = st.text_input("KDV", value="" if kdv is None else str(kdv))
            t_in = st.text_input("Toplam", value="" if top is None else str(top))
            if st.button("Kaydet", key="save_edit"):
                summ = dict(fields)
                summ["net_tutar"] = parse_tr_number(n_in)
                summ["kdv_tutari"] = parse_tr_number(k_in)
                summ["genel_toplam"] = parse_tr_number(t_in)
                cur["summary"] = summ
                hid = st.session_state.get("history")
                if isinstance(hid, list):
                    for i, r in enumerate(hid):
                        if isinstance(r, dict) and r.get("id") == cur.get("id"):
                            hid[i] = cur
                            break
                st.session_state["history"] = hid
                st.rerun()
    else:
        st.markdown(
            """
            <div class="empty-audit-box">
                <p class="t1">Analiz bekleniyor</p>
                <p class="t2">Soldan fatura yükleyin ve <strong>Analiz et</strong> seçin.
                Çıkarılan tutarlar, doğrulama ve yapay zekâ özeti burada görünür.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if valid_hist:
    with st.expander("Oturum denetim günlüğü", expanded=False):
        log_rows = []
        for r in reversed(valid_hist):
            summ = r.get("summary") if isinstance(r.get("summary"), dict) else {}
            log_rows.append(
                {
                    "Dosya": r.get("filename"),
                    "Kimlik": r.get("id"),
                    "Kategori": r.get("ai_category_tr"),
                    "Toplam (TL)": summ.get("genel_toplam"),
                    "Net": summ.get("net_tutar"),
                    "KDV": summ.get("kdv_tutari"),
                }
            )
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

        _stamp = _export_timestamp()
        _rows = _audit_rows_for_export(valid_hist)
        _df_exp = pd.DataFrame(_rows)
        _buf_csv = io.BytesIO()
        _df_exp.to_csv(_buf_csv, index=False, encoding="utf-8-sig")
        _csv_bytes = _buf_csv.getvalue()
        _json_bytes = json.dumps(
            _audit_json_payload(valid_hist),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        st.caption("Denetim izi: oturumdaki kayıtları indirin (Excel ile CSV açılabilir).")
        ex1, ex2 = st.columns(2, gap="small")
        with ex1:
            st.download_button(
                "📥 CSV indir",
                data=_csv_bytes,
                file_name=f"auditsight_denetim_{_stamp}.csv",
                mime="text/csv",
                use_container_width=True,
                key="export_audit_csv",
            )
        with ex2:
            st.download_button(
                "📥 JSON indir",
                data=_json_bytes,
                file_name=f"auditsight_denetim_{_stamp}.json",
                mime="application/json",
                use_container_width=True,
                key="export_audit_json",
            )

with st.container():
    fc1, fc2, fc3 = st.columns([1.05, 1.15, 1.0], gap="medium")
    with fc1:
        st.markdown(
            '<h3 class="footer-block" style="margin:0;font-size:1.05rem;">© 2026 AuditSight Pro</h3>'
            '<p class="footer-sub footer-block">Developed by Fatmanur Yılmaz</p>',
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown('<p class="footer-heading footer-block">Geliştirici ile iletişim</p>', unsafe_allow_html=True)
        b1, b2 = st.columns(2, gap="small")
        with b1:
            st.link_button("🔗 LinkedIn", _li_url, use_container_width=True, type="secondary")
        with b2:
            st.link_button("👩‍💻 GitHub", _gh_url, use_container_width=True, type="secondary")
    with fc3:
        st.markdown(
            '<p class="footer-heading footer-block">Program bilgisi</p>'
            '<p class="footer-block footer-sub" style="margin:0.2rem 0 0 0;">YGA &amp; UP School — Future Talent Program</p>'
            '<p class="footer-block footer-sub" style="margin:0.35rem 0 0 0;opacity:0.9;">Yapay zekâ modülü · Gemini 1.5 Flash</p>',
            unsafe_allow_html=True,
        )
