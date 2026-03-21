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

_NUM = r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2}"
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
                "label": "⚠️ Net+KDV=Toplam; KDV oranı standart değil.",
                "color": "orange",
            })
        else:
            result.update({
                "status": "RISK",
                "label": "🚨 Altın kural başarısız.",
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
    Fatura alanlarını çıkarır.
    """
    lines_y = _parse_ocr_lines(ocr_lines)
    if not lines_y:
        lines_y = _lines_from_text(text)
    
    net = 0.0
    kdv = 0.0
    total = None
    
    # ============ 1. KDV MATRAHI (NET) ============
    for ln in lines_y:
        tx = ln["text"]
        tx_lower = tx.lower()
        
        if "kdv matrah" in tx_lower or "matrah" in tx_lower:
            numbers = _MONEY_IN_LINE.findall(tx)
            if numbers:
                for num in numbers:
                    val = parse_tr_number(num)
                    if val and val > 0:
                        net = val
                        break
            
            if net == 0:
                idx = lines_y.index(ln)
                for j in range(idx + 1, min(idx + 3, len(lines_y))):
                    next_tx = lines_y[j]["text"]
                    numbers = _MONEY_IN_LINE.findall(next_tx)
                    if numbers:
                        for num in numbers:
                            val = parse_tr_number(num)
                            if val and val > 0:
                                net = val
                                break
                    if net > 0:
                        break
        
        if net > 0:
            break
    
    # ============ 2. HESAPLANAN KDV ============
    for ln in lines_y:
        tx = ln["text"]
        tx_lower = tx.lower()
        
        if "hesaplanan kdv" in tx_lower or ("hesaplanan" in tx_lower and "kdv" in tx_lower):
            numbers = _MONEY_IN_LINE.findall(tx)
            if numbers:
                for num in numbers:
                    val = parse_tr_number(num)
                    if val and val > 0:
                        kdv = val
                        break
            
            if kdv == 0:
                idx = lines_y.index(ln)
                for j in range(idx + 1, min(idx + 3, len(lines_y))):
                    next_tx = lines_y[j]["text"]
                    numbers = _MONEY_IN_LINE.findall(next_tx)
                    if numbers:
                        for num in numbers:
                            val = parse_tr_number(num)
                            if val and val > 0:
                                kdv = val
                                break
                    if kdv > 0:
                        break
        
        if kdv > 0:
            break
    
    # ============ 3. ÖDENECEK TUTAR (GENEL TOPLAM) ============
    for ln in lines_y:
        tx = ln["text"]
        tx_lower = tx.lower()
        
        if "ödenecek tutar" in tx_lower or "ödenecek" in tx_lower or "vergiler dahil" in tx_lower:
            numbers = _MONEY_IN_LINE.findall(tx)
            if numbers:
                for num in numbers:
                    val = parse_tr_number(num)
                    if val and val > 0:
                        total = val
                        break
            
            if total is None:
                idx = lines_y.index(ln)
                for j in range(idx + 1, min(idx + 3, len(lines_y))):
                    next_tx = lines_y[j]["text"]
                    numbers = _MONEY_IN_LINE.findall(next_tx)
                    if numbers:
                        for num in numbers:
                            val = parse_tr_number(num)
                            if val and val > 0:
                                total = val
                                break
                    if total is not None:
                        break
        
        if total is not None:
            break
    
    # ============ 4. ALTERNATİF: TÜM SAYILARI TOPLA ============
    if net == 0 or kdv == 0 or total is None:
        all_numbers = []
        for ln in lines_y:
            numbers = _MONEY_IN_LINE.findall(ln["text"])
            for num in numbers:
                val = parse_tr_number(num)
                if val and 0 < val < 1000000:
                    all_numbers.append(val)
        
        unique_numbers = sorted(set(all_numbers))
        
        if len(unique_numbers) >= 3:
            if total is None:
                total = unique_numbers[-1]
            
            candidates = [n for n in unique_numbers if n < total]
            if candidates and net == 0:
                net = max(candidates)
            
            if net > 0 and kdv == 0:
                kdv_candidates = [n for n in unique_numbers if n < net and n > 0]
                if kdv_candidates:
                    kdv = max(kdv_candidates)
    
    # ============ 5. EKSİK ALANLARI TAMAMLA ============
    if net > 0 and kdv > 0 and total is None:
        total = float(_to_decimal_2(net) + _to_decimal_2(kdv))
    
    if net > 0 and total and total > net and kdv == 0:
        kdv = float(_to_decimal_2(total) - _to_decimal_2(net))
    
    if kdv > 0 and total and total > kdv and net == 0:
        net = float(_to_decimal_2(total) - _to_decimal_2(kdv))
    
    # ============ 6. SON ÇARE ============
    if total is None:
        bottom_first = _lines_bottom_first(lines_y)
        total = _extract_last_currency_value(bottom_first)
    
    # ============ 7. KDV ORANINI KONTROL ET VE DÜZELT ============
    if net > 0 and kdv > 0:
        expected_kdv = net * 0.20
        if abs(kdv - expected_kdv) > 1.0:
            kdv = round(net * 0.20, 2)
            total = net + kdv
    
    return {
        "firma_adi": "Tespit Edilemedi",
        "tarih": datetime.now().strftime("%Y-%m-%d"),
        "net_tutar": net,
        "kdv_tutari": kdv,
        "genel_toplam": total if total is not None else 0.0,
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
