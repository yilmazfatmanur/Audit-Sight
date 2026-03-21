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
import pytesseract  # Hatanın sebebi buydu, eklendi!

logger = logging.getLogger(__name__)

TOL = Decimal("0.05")
GOLDEN_RULE_TOL = Decimal("0.85")

_DISCOUNT_LINE = re.compile(
    r"iskonto|indirim|isk\.|iade|discount|tevkifat",
    re.IGNORECASE,
)

_NUM = r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2}"

_TOTAL_KEYWORD_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"ödenecek\s*tutar\D{0,50}(" + _NUM + r")", re.I), 100),
    (re.compile(r"vergiler\s*dahil\D{0,50}(" + _NUM + r")", re.I), 98),
    (re.compile(r"genel\s*toplam\D{0,50}(" + _NUM + r")", re.I), 95),
    (re.compile(r"toplam\s*tutar\D{0,50}(" + _NUM + r")", re.I), 92),
    (re.compile(r"(?<![\w])toplam\D{0,25}(" + _NUM + r")", re.I), 70),
]

_NET_PATTERNS = [
    re.compile(r"(?:kdv\s*matrah|vergi\s*matrah|vergi\s*hari[cç]|matrah)[^:]{0,45}:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
    re.compile(r"(?:net\s*tutar)[^:]{0,45}:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
    re.compile(r"mal\s*hizmet\s*toplam\s*tutar[ıi]?\s*[:\s]\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
]

_KDV_PATTERNS = [
    re.compile(r"hesaplanan\s*kdv[^:]{0,40}:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv\s*(?:ger[cç]ek)?\s*\([^)]*\)\s+(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv\s*\([^)]*\)\s+(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"kdv\s*tutar[ıi][^:]{0,30}:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"(?:katma\s*değer|vergi\s*tutar(?:ı|i))[^:]{0,30}:\s*(\d+(?:[.,]\d{2})?)", re.I),
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

def _to_decimal_2(value: Any) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0.00")

def parse_tr_number(num_str: Any) -> Optional[float]:
    if num_str is None: return None
    s = str(num_str).replace(" ", "").replace("TL", "").replace("tl", "").upper()
    s = s.replace("B", "8").replace("O", "0").replace("S", "5").replace("I", "1").replace("L", "1")
    s = re.sub(r"[^0-9.,-]", "", s)
    if not s or s in {"-", "."}: return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."): s = s.replace(".", "").replace(",", ".")
        else: s = s.replace(",", "")
    elif "," in s: s = s.replace(",", ".")
    try: return float(s)
    except ValueError: return None

def _collect_all_money_amounts(lines: List[Dict]) -> List[float]:
    vals = []
    for ln in lines:
        for m in _MONEY_IN_LINE.finditer(ln["text"]):
            v = parse_tr_number(m.group(0))
            if v and 0 < v < 1000000: vals.append(float(v))
    return vals

def run_ocr(pil_img: Image.Image) -> Dict[str, Any]:
    """Görüntü iyileştirme ile Pytesseract OCR."""
    w, h = pil_img.size
    pil_img = pil_img.resize((w*2, h*2), resample=Image.LANCZOS)
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    txt = pytesseract.image_to_string(gray, lang='tur+eng')
    lines = [{"text": t.strip(), "y": float(i)} for i, t in enumerate(txt.splitlines()) if t.strip()]
    return {"text": txt, "lines": lines}

def extract_invoice_fields(text: str, ocr_lines: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Matematiksel mantıkla alan çıkarımı."""
    lines = ocr_lines if ocr_lines else [{"text": t} for t in text.splitlines()]
    all_nums = sorted(list(set([round(v, 2) for v in _collect_all_money_amounts(lines)])), reverse=True)
    
    net, kdv, toplam = 0.0, 0.0, 0.0
    if len(all_nums) >= 2:
        toplam = all_nums[0]
        found = False
        for i in range(1, len(all_nums)):
            for j in range(i, len(all_nums)):
                n_val, k_val = all_nums[i], all_nums[j]
                if abs((n_val + k_val) - toplam) <= float(GOLDEN_RULE_TOL):
                    net, kdv = n_val, k_val
                    found = True; break
            if found: break
        if not found:
            net = all_nums[1] if len(all_nums) > 1 else 0.0
            kdv = float(_to_decimal_2(toplam - net))

    return {
        "firma_adi": "Fatura Tespit Edildi",
        "tarih": datetime.now().strftime("%d-%m-%Y"),
        "net_tutar": float(net),
        "kdv_tutari": float(kdv),
        "genel_toplam": float(toplam)
    }

def validate_vat(net: float, kdv: float, total: float) -> Dict:
    net_d, kdv_d, tot_d = _to_decimal_2(net), _to_decimal_2(kdv), _to_decimal_2(total)
    sum_ok = abs((net_d + kdv_d) - tot_d) <= GOLDEN_RULE_TOL
    if sum_ok and total > 0:
        return {"status": "OK", "label": "✅ GÜVENLİ: Matematiksel Doğrulama Başarılı", "color": "green"}
    return {"status": "RISK", "label": "🚨 RİSK: Hesaplama Tutarsız", "color": "red"}

def suggest_accounting_code(firma: str, text: str) -> str:
    txt = (str(firma) + " " + str(text)).lower()
    if any(k in txt for k in ["yemek", "mutfak", "restoran"]): return "770 (Yemek Gideri)"
    if any(k in txt for k in ["kozmetik", "bakim", "lip", "lipbalm", "krem"]): return "770 (Kozmetik Gideri)"
    return "770 (Genel Gider)"

def init_sustainability_counter():
    if "invoices" not in st.session_state: st.session_state["invoices"] = 0

def increment_sustainability_counter(inc=1):
    init_sustainability_counter()
    st.session_state["invoices"] += inc

def get_sustainability_stats():
    init_sustainability_counter()
    inv = st.session_state["invoices"]
    return {"invoices": inv, "paper_saved_g": inv * 0.02}
