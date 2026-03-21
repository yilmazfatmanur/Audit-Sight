# Audit-Sight/utils.py
"""
AuditSight Pro — OCR (EasyOCR), sustainability stats, VAT validation.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

TOL = Decimal("0.05")
GOLDEN_RULE_TOL = Decimal("0.50")

_DISCOUNT_LINE = re.compile(
    r"iskonto|indirim|isk\.|iade|discount|tevkifat",
    re.IGNORECASE,
)

# Binlik ayraçlı sayı: 1.950,00 veya 1,950.00 veya 763,00
_NUM = r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2}"

_TOTAL_KEYWORD_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"ödenecek\s*tutar\D{0,50}(" + _NUM + r")", re.I), 100),
    (re.compile(r"vergiler\s*dahil\D{0,50}(" + _NUM + r")", re.I), 98),
    (re.compile(r"genel\s*toplam\D{0,50}(" + _NUM + r")", re.I), 95),
    (re.compile(r"toplam\s*tutar\D{0,50}(" + _NUM + r")", re.I), 92),
    (re.compile(r"(?<![\w])toplam\D{0,25}(" + _NUM + r")", re.I), 70),
]

# NET patterns - REMOVED "mal hizmet toplam" to avoid wrong picks
_NET_PATTERNS = [
    re.compile(r"(?:kdv\s*matrahı?|vergi\s*matrahı?|matrah)[^:]{0,45}:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
    re.compile(r"(?:net\s*tutar)[^:]{0,45}:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
]

_KDV_PATTERNS = [
    re.compile(r"hesaplanan\s*kdv\s*\([^)]*\)\s*:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv\s*\([^)]*\)\s+(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv[^:]{0,40}:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"kdv\s*tutar[ıi][^:]{0,30}:\s*(\d+(?:[.,]\d{2})?)", re.I),
]

_KDV_20_PATTERNS = [
    re.compile(r"hesaplanan\s*kdv\s*\(%20[.,]?\d*\)\s*:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv\s*\(%20[.,]?\d*\)\s+(\d+(?:[.,]\d{2})?)", re.I),
]

_MONEY_IN_LINE = re.compile(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2}")

_PRODUCT_ROW_HINT = re.compile(r"\badet\b|\bpcs\b|\bbirim\s*fiyat\b", re.I)


@dataclass
class OcrLine:
    text: str
    confidence: float
    y_center: float = 0.0


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_lower(s: str) -> str:
    return (s or "").lower()


def _to_decimal_2(value: Any) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0.00")


def parse_tr_number(num_str: Any) -> Optional[float]:
    if num_str is None:
        return None
    s = str(num_str).replace(" ", "").replace("TL", "").replace("tl", "")
    s = s.replace("B", "8").replace("O", "0").replace("S", "5") \
         .replace("I", "1").replace("l", "1").replace("G", "6")
    s = re.sub(r"[^0-9.,-]", "", s)
    if not s or s in {"-", "."}:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _is_discount_line(text: str) -> bool:
    return bool(_DISCOUNT_LINE.search(_safe_lower(text or "")))


def _parse_ocr_lines(ocr_lines: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(ocr_lines, list):
        return out
    for item in ocr_lines:
        if not isinstance(item, dict):
            continue
        tx = _normalize_whitespace(str(item.get("text") or ""))
        if not tx:
            continue
        y = item.get("y_center", item.get("y", 0.0))
        try:
            yf = float(y)
        except (TypeError, ValueError):
            yf = 0.0
        out.append({"text": tx, "y": yf})
    out.sort(key=lambda x: x["y"])
    return out


def _lines_from_text(text: str) -> List[Dict[str, Any]]:
    parts = [p.strip() for p in (text or "").splitlines() if p.strip()]
    return [{"text": p, "y": float(i)} for i, p in enumerate(parts)]


def _lines_bottom_first(lines_y_sorted_asc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(lines_y_sorted_asc, key=lambda x: x["y"], reverse=True)


def _collect_pattern_amounts(
    lines_y_asc: List[Dict[str, Any]], patterns: List[re.Pattern], *, skip_product_rows: bool = True
) -> List[Tuple[float, float]]:
    found: List[Tuple[float, float]] = []
    for ln in lines_y_asc:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        if skip_product_rows and bool(_PRODUCT_ROW_HINT.search(tx)):
            continue
        for pat in patterns:
            m = pat.search(tx)
            if not m:
                continue
            v = parse_tr_number(m.group(1))
            if v is not None and 0 < v < 50_000_000:
                found.append((float(v), float(ln["y"])))
                break
    return found


def _grand_total_candidates(lines_bottom_first: List[Dict[str, Any]]) -> List[Tuple[float, int, float]]:
    cands: List[Tuple[float, int, float]] = []
    seen: set = set()

    for ln in lines_bottom_first:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        if bool(_PRODUCT_ROW_HINT.search(tx)):
            continue
        y = float(ln["y"])
        for pat, pri in _TOTAL_KEYWORD_PATTERNS:
            m = pat.search(tx)
            if not m:
                continue
            v = parse_tr_number(m.group(1))
            if v is None or v <= 0:
                continue
            key = (round(v, 2), pri)
            if key in seen:
                continue
            seen.add(key)
            cands.append((float(v), int(pri), y))

    cands.sort(key=lambda t: (-t[1], -t[2]))
    return cands


def _extract_last_currency_value(lines_bottom_first: List[Dict[str, Any]]) -> Optional[float]:
    for ln in lines_bottom_first:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        if bool(_PRODUCT_ROW_HINT.search(tx)):
            continue
        matches = list(_MONEY_IN_LINE.finditer(tx))
        if not matches:
            continue
        for m in reversed(matches):
            v = parse_tr_number(m.group(0))
            if v is not None and 0 < v < 50_000_000:
                return float(v)
    return None


def init_sustainability_counter(*, grams_per_invoice: float = 0.02) -> None:
    if "sustainability_invoices" not in st.session_state:
        st.session_state["sustainability_invoices"] = 0
    if "sustainability_paper_saved_g" not in st.session_state:
        st.session_state["sustainability_paper_saved_g"] = 0.0
    if "sustainability_grams_per_invoice" not in st.session_state:
        st.session_state["sustainability_grams_per_invoice"] = float(grams_per_invoice)


def increment_sustainability_counter(increment_by: int = 1) -> None:
    init_sustainability_counter()
    inc = max(int(increment_by or 0), 0)
    st.session_state["sustainability_invoices"] = int(
        st.session_state.get("sustainability_invoices", 0)
    ) + inc
    gpi = float(st.session_state.get("sustainability_grams_per_invoice", 0.02))
    prev = float(st.session_state.get("sustainability_paper_saved_g", 0.0))
    st.session_state["sustainability_paper_saved_g"] = float(
        _to_decimal_2(Decimal(str(prev)) + Decimal(str(inc * gpi)))
    )


def get_sustainability_stats() -> Dict[str, float]:
    init_sustainability_counter()
    return {
        "invoices": float(st.session_state.get("sustainability_invoices", 0)),
        "paper_saved_g": float(st.session_state.get("sustainability_paper_saved_g", 0.0)),
    }


def validate_vat(
    net_tutar: Optional[float],
    kdv_tutari: Optional[float],
    genel_toplam: Optional[float] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "UNKNOWN",
        "label": "Veri Bekleniyor...",
        "color": "gray",
    }
    try:
        def _to_opt(x: Any) -> Optional[Decimal]:
            if x is None:
                return None
            if isinstance(x, str) and not str(x).strip():
                return None
            return _to_decimal_2(x)

        net_d = _to_opt(net_tutar)
        kdv_d = _to_opt(kdv_tutari)
        total_d = _to_opt(genel_toplam)

        if total_d is None and net_d is not None and kdv_d is not None:
            total_d = _to_decimal_2(net_d + kdv_d)

        if net_d is None or total_d is None:
            return result

        kdv_eff = kdv_d if kdv_d is not None else Decimal("0.00")
        sum_ok = abs((net_d + kdv_eff) - total_d) <= GOLDEN_RULE_TOL

        matched_rate: Optional[float] = None
        if net_d > 0 and kdv_eff > 0:
            actual_rate = kdv_eff / net_d
            for r in (Decimal("0.20"), Decimal("0.10"), Decimal("0.01")):
                if abs(actual_rate - r) < Decimal("0.02"):
                    matched_rate = float(r)
                    break

        if sum_ok and matched_rate is not None:
            pct = int(round(matched_rate * 100))
            result.update({
                "status": "OK",
                "label": f"✅ GÜVENLİ: Altın kural sağlandı; KDV oranı ~%{pct}",
                "color": "green",
            })
        elif sum_ok:
            result.update({
                "status": "ŞÜPHELİ",
                "label": f"⚠️ Net+KDV=Toplam (±{float(GOLDEN_RULE_TOL):.2f}); KDV oranı standart değil.",
                "color": "orange",
            })
        else:
            diff = abs((net_d + kdv_eff) - total_d)
            result.update({
                "status": "RISK",
                "label": f"🚨 Altın kural başarısız: Net+KDV ≠ Toplam (|fark|={float(diff):.2f}).",
                "color": "red",
            })
    except Exception as exc:
        logger.exception("validate_vat: %s", exc)
        result.update({
            "status": "ERROR",
            "label": "Denetim sırasında hata; verileri kontrol edin.",
            "color": "gray",
        })
    return result


def suggest_accounting_code(firma_adi: Any = None, text: Any = None) -> str:
    try:
        haystack = (_safe_lower(str(firma_adi)) + " " + _safe_lower(str(text))).strip()
        if not haystack or haystack == "none none":
            return "Belirlenemedi (Manuel)"
        rules = [
            (["yemek", "restoran", "lokanta", "gida", "mutfak"], "770 (Yemek Gideri)"),
            (["yakit", "petrol", "shell", "opet", "akaryakit"], "760 (Pazarlama/Nakliye)"),
            (["kirtasiye", "ofis", "kagit", "copy"], "770 (Ofis Gideri)"),
            (["konaklama", "otel", "pansiyon"], "770 (Seyahat Gideri)"),
            (["bilgisayar", "yazilim", "lisans", "apple"], "255 (Demirbaşlar)"),
            (["kozmetik", "kremi", "bakim", "lip", "parfum"], "770 (Kozmetik Gideri)"),
        ]
        for keywords, code in rules:
            if any(k in haystack for k in keywords):
                return code
        return "770 (Genel Gider)"
    except Exception:
        return "770 (Genel Gider)"


@st.cache_resource(show_spinner=False)
def load_easyocr_reader():
    import easyocr
    return easyocr.Reader(["tr", "en"], gpu=False)


def run_ocr(pil_img: Image.Image) -> Dict[str, Any]:
    """EasyOCR ile fatura okuma."""
    reader = load_easyocr_reader()
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    raw = reader.readtext(gray, detail=1)
    lines: List[OcrLine] = []
    for item in raw:
        try:
            bbox = item[0]
            txt = _normalize_whitespace(str(item[1]))
            conf = float(item[2])
            ys = [float(p[1]) for p in bbox]
            y_c = sum(ys) / len(ys) if ys else 0.0
        except (TypeError, IndexError, ValueError, KeyError):
            txt = _normalize_whitespace(str(item[1]) if len(item) > 1 else "")
            conf = float(item[2]) if len(item) > 2 else 0.0
            y_c = 0.0
        if txt:
            lines.append(OcrLine(text=txt, confidence=conf, y_center=y_c))
    lines.sort(key=lambda ln: ln.y_center)
    return {
        "text": "\n".join(ln.text for ln in lines),
        "lines": [ln.__dict__ for ln in lines],
    }


def extract_invoice_fields(
    text: str, ocr_lines: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Iskontolu faturalar için doğru Net/KDV/Toplam tespiti.
    Öncelik: KDV Matrahı (net) + Hesaplanan KDV (kdv) + Ödenecek Tutar (toplam)
    """
    lines_y = _parse_ocr_lines(ocr_lines)
    if not lines_y:
        lines_y = _lines_from_text(text)

    bottom_first = _lines_bottom_first(lines_y)

    net = 0.0
    kdv = 0.0
    genel_toplam = None

    # ---- 1. KDV MATRAHI (NET) ----
    net_from_matrah = None
    # Aynı satırda ara
    for ln in lines_y:
        tx = _normalize_whitespace(ln["text"])
        # Önce "KDV Matrahı (%20.00): 404,84" gibi
        m = re.search(r"kdv\s*matrahı?\s*\([^)]*\)\s*:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx, re.I)
        if not m:
            m = re.search(r"kdv\s*matrahı?\s*:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx, re.I)
        if m:
            net_from_matrah = parse_tr_number(m.group(1))
            if net_from_matrah:
                break

    # Bulunamadıysa, "KDV Matrahı" içeren satırdan sonraki satırda sayı ara
    if not net_from_matrah:
        for i, ln in enumerate(lines_y):
            if re.search(r"kdv\s*matrahı?", ln["text"], re.I):
                # sonraki satır(lar)da sayı ara
                for j in range(i+1, min(i+4, len(lines_y))):
                    next_tx = lines_y[j]["text"]
                    m = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", next_tx)
                    if m:
                        net_from_matrah = parse_tr_number(m.group(1))
                        if net_from_matrah:
                            break
                if net_from_matrah:
                    break

    # ---- 2. HESAPLANAN KDV ----
    kdv_from_calc = None
    for ln in lines_y:
        tx = _normalize_whitespace(ln["text"])
        # Aynı satırda ara
        m = re.search(r"hesaplanan\s*kdv\s*\([^)]*\)\s*:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx, re.I)
        if not m:
            m = re.search(r"hesaplanan\s*kdv\s*:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx, re.I)
        if m:
            kdv_from_calc = parse_tr_number(m.group(1))
            if kdv_from_calc:
                break

    # Bulunamadıysa, "Hesaplanan KDV" içeren satırdan sonraki satırda sayı ara
    if not kdv_from_calc:
        for i, ln in enumerate(lines_y):
            if re.search(r"hesaplanan\s*kdv", ln["text"], re.I):
                for j in range(i+1, min(i+4, len(lines_y))):
                    next_tx = lines_y[j]["text"]
                    m = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", next_tx)
                    if m:
                        kdv_from_calc = parse_tr_number(m.group(1))
                        if kdv_from_calc:
                            break
                if kdv_from_calc:
                    break

    # ---- 3. ÖDENECEK TUTAR (GENEL TOPLAM) ----
    total_from_payable = None
    for ln in lines_y:
        tx = _normalize_whitespace(ln["text"])
        m = re.search(r"ödenecek\s*tutar\s*:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx, re.I)
        if not m:
            m = re.search(r"vergiler\s*dahil\s*toplam\s*tutarı?\s*:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx, re.I)
        if m:
            total_from_payable = parse_tr_number(m.group(1))
            if total_from_payable:
                break

    # Bulunamadıysa, "Ödenecek Tutar" içeren satırdan sonraki satırda sayı ara
    if not total_from_payable:
        for i, ln in enumerate(lines_y):
            if re.search(r"ödenecek\s*tutar|vergiler\s*dahil", ln["text"], re.I):
                for j in range(i+1, min(i+4, len(lines_y))):
                    next_tx = lines_y[j]["text"]
                    m = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", next_tx)
                    if m:
                        total_from_payable = parse_tr_number(m.group(1))
                        if total_from_payable:
                            break
                if total_from_payable:
                    break

    # ---- 4. EŞLEŞTİRME ----
    if net_from_matrah and kdv_from_calc and total_from_payable:
        # Doğrula: net + kdv ≈ total
        net_dec = _to_decimal_2(net_from_matrah)
        kdv_dec = _to_decimal_2(kdv_from_calc)
        total_dec = _to_decimal_2(total_from_payable)
        if abs(net_dec + kdv_dec - total_dec) <= Decimal("1.00"):
            net = net_from_matrah
            kdv = kdv_from_calc
            genel_toplam = total_from_payable
            logger.info(f"Context extraction SUCCESS: Net={net}, KDV={kdv}, Total={genel_toplam}")

    # Eksik varsa, elimizdeki verilerle tamamla
    if net == 0.0 and net_from_matrah:
        net = net_from_matrah
    if kdv == 0.0 and kdv_from_calc:
        kdv = kdv_from_calc
    if genel_toplam is None and total_from_payable:
        genel_toplam = total_from_payable

    # Net + KDV varsa, toplamı hesapla
    if net > 0 and kdv > 0 and genel_toplam is None:
        genel_toplam = float(_to_decimal_2(net) + _to_decimal_2(kdv))

    # Net ve toplam varsa, KDV'yi hesapla
    if kdv == 0.0 and net > 0 and genel_toplam and genel_toplam > net:
        kdv = float(_to_decimal_2(genel_toplam) - _to_decimal_2(net))

    # Toplam ve KDV varsa, net'i hesapla
    if net == 0.0 and kdv > 0 and genel_toplam and genel_toplam > kdv:
        net = float(_to_decimal_2(genel_toplam) - _to_decimal_2(kdv))

    # Hala net yoksa, genel toplam adaylarından en büyüğü toplam olsun
    if genel_toplam is None:
        candidates = _grand_total_candidates(bottom_first)
        if candidates:
            genel_toplam = candidates[0][0]

    # Son çare: en alttaki sayı
    if genel_toplam is None:
        last_val = _extract_last_currency_value(bottom_first)
        if last_val:
            genel_toplam = last_val

    # Net hala 0 ise, KDV Matrahı veya diğer mantıklı değerleri dene
    if net == 0.0 and genel_toplam:
        # KDV Matrahı bulunamamışsa, toplamın %16.67'si (1/6) KDV olabilir (standart %20 için)
        # Ama bu yanıltıcı olabilir, sadece fallback olarak
        # Önce KDV Matrahı için tekrar dene
        if not net_from_matrah:
            for ln in lines_y:
                if re.search(r"matrah", ln["text"], re.I):
                    m = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", ln["text"])
                    if m:
                        net = parse_tr_number(m.group(1))
                        if net:
                            break
        if net == 0.0:
            # Hiç net bulunamadı, varsayılan olarak toplamın %83.33'ünü net al (KDV %20 ise)
            net = float(_to_decimal_2(genel_toplam) / Decimal("1.20"))
            kdv = float(_to_decimal_2(genel_toplam) - _to_decimal_2(net))

    logger.info(f"FINAL: Net={net}, KDV={kdv}, Total={genel_toplam}")

    return {
        "firma_adi": "Tespit Edilemedi",
        "tarih": datetime.now().strftime("%Y-%m-%d"),
        "net_tutar": float(net) if net is not None else 0.0,
        "kdv_tutari": float(kdv) if kdv is not None else 0.0,
        "genel_toplam": float(genel_toplam) if genel_toplam is not None else None,
        "total_override_applied": False,
        "total_source": "context_based",
    }


def build_extracted_summary(ocr_text: str, ocr_lines: list, fields: dict) -> dict:
    if not isinstance(fields, dict):
        fields = {}
    return {
        "fields": dict(fields),
        "accounting_code": suggest_accounting_code(
            fields.get("firma_adi"), str(ocr_text or "")
        ),
        "vat_validation": validate_vat(
            fields.get("net_tutar"),
            fields.get("kdv_tutari"),
            fields.get("genel_toplam"),
        ),
        "ocr_lines": ocr_lines if isinstance(ocr_lines, list) else [],
    }
