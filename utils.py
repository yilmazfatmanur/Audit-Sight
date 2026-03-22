# Audit-Sight/utils.py
"""AuditSight Pro MVP+ — Etiket tabanlı Net/KDV/Ödenecek, iskonto, Tesseract/EasyOCR, Gemini notu."""
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

GOLDEN_RULE_TOL = Decimal("0.50")
# Satır KDV + matrah ile ödenecek tutar eşlemesi (yuvarlama / OCR gürültüsü)
_KDV_LINE_SUM_TOL = Decimal("1.20")
# Uyumluluk kartı: çıkarım ile aynı eşik (0,50 ile çelişmesin)
GOLDEN_RULE_VALIDATION_TOL = max(GOLDEN_RULE_TOL, _KDV_LINE_SUM_TOL)

_NUM = r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2}"
# TL tutarları; parantez içi %20.00 gibi oranlar ayrı maskelenir
_MONEY_TOKEN_RE = re.compile(r"(?:\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})")
# Sadece KDV oranı: (%20.00), (% 1,00) — tutar regex'ine girmesin
_RATE_IN_PARENS = re.compile(r"\(\s*%[\d\s.,]+\s*\)", re.I)

# Ürün satırı — etiketli tutar aramasında atla
_PRODUCT_ROW_HINT = re.compile(r"\badet\b|\bpcs\b|\bbirim\s*fiyat\b|\bqty\b", re.I)
_SKIP_FOR_TOTAL = re.compile(r"ara\s*toplam|subtotal|birim\s*fiyat", re.I)
# KDV satırı sanılırken ödenecek / genel toplam satırlarını ele
_KDV_LINE_EXCLUDE = re.compile(
    r"ö?denecek\s*tutar|amount\s*payable|balance\s*due|"
    r"vergiler\s*dahil\s*toplam|genel\s*toplam|grand\s*total",
    re.I,
)

# İskonto / iade satırları — ödenecek tutar sanılmasın
_DISCOUNT_LINE = re.compile(
    r"iskonto|indirim|isk\.|iade|discount|tevkifat|tevkif",
    re.IGNORECASE,
)


def _is_discount_only_line(tx: str) -> bool:
    """Ödenecek / genel toplam içermiyorsa ve indirim satırı gibi görünüyorsa toplam adayı değil."""
    if not tx or not _DISCOUNT_LINE.search(tx):
        return False
    if re.search(
        r"ö?denecek|vergiler\s*dahil|genel\s*toplam|amount\s*payable|balance\s*due",
        tx,
        re.I,
    ):
        return False
    return True

# --- Net: önce KDV matrahı / vergi hariç (iskonto sonrası), sonra mal+hizmet toplam ---
_NET_STRICT_PATTERNS: List[re.Pattern] = [
    # Oran parantezi (%20.00) atlanır; yalnızca sonrasındaki TL tutarı
    re.compile(
        r"kdv\s*matrah[ıi]?(?:\s*\(\s*%[^)]*\))?\s*(?:\(\s*(?!%)[^)]*\)\s*)?\s*[:\s]+("
        + _NUM
        + r")",
        re.I,
    ),
    # İskontolu / indirim sonrası matrah satırları
    re.compile(
        r"kdv\s*matrah[ıi]?\s*\([^)]*(?:iskonto|indirim)[^)]*\)\s*[:\s]+(" + _NUM + r")",
        re.I,
    ),
    re.compile(r"(?:vat\s*base|tax\s*base|taxable\s*amount)\D{0,40}(" + _NUM + r")", re.I),
    re.compile(r"(?:kdv\s*matrah|vergi\s*matrah[ıi]?|vergi\s*hari[cç]\s*tutar)[^:]{0,45}:\s*(" + _NUM + r")", re.I),
    re.compile(r"(?:vergi\s*hari[cç]|excluding\s*tax|excl\.?\s*vat)\D{0,40}(" + _NUM + r")", re.I),
    re.compile(r"(?:net\s*tutar|net\s*amount)\D{0,35}:\s*(" + _NUM + r")", re.I),
]

_NET_FALLBACK_PATTERNS: List[re.Pattern] = [
    re.compile(r"mal\s*hizmet\s*toplam\s*tutar[ıi]?\D{0,30}(" + _NUM + r")", re.I),
    re.compile(r"mal\s*hizmet\s*toplam\D{0,20}:\s*(" + _NUM + r")", re.I),
    re.compile(r"(?:goods\s*(?:and|&)\s*services|line\s*items?\s*total)\D{0,40}(" + _NUM + r")", re.I),
]

# --- KDV: yalnızca hesaplanan / KDV tutarı etiketi ---
_KDV_PATTERNS: List[re.Pattern] = [
    # Grup 1: TL tutarı — % veya oran parantezi yakalamaz
    re.compile(
        r"hesaplanan\s*kdv(?:\s*\([^)]*\))?\s*[:\s]+\s*(" + _NUM + r")",
        re.I,
    ),
    re.compile(
        r"hesaplanan\s*kdv(?:\s*\([^)]*\))?\s+(" + _NUM + r")(?=\s*$|\s*TL|\s*₺)",
        re.I,
    ),
    # %1 / % 1,00 gibi düşük oranlı satırlar: tutar genelde parantezden hemen sonra
    re.compile(
        r"hesaplanan\s*kdv\s*\(\s*%?\s*1[\s.,]*\d*\s*\)\s*[:\s]*(" + _NUM + r")",
        re.I,
    ),
    re.compile(
        r"hesaplanan\s*kdv\s*\(\s*%?\s*1[\s.,]*\d*\s*\)\s+(" + _NUM + r")(?=\s*TL|\s*₺|\s*$)",
        re.I,
    ),
    re.compile(r"hesaplanan\s*kdv[^%]{0,50}:\s*(" + _NUM + r")", re.I),
    re.compile(r"(?:calculated\s*vat|vat\s*amount)\D{0,40}:\s*(" + _NUM + r")", re.I),
    re.compile(
        r"kdv\s*tutar[ıi]?(?:\s*\([^)]*\))?\s*[:\s]+\s*(" + _NUM + r")",
        re.I,
    ),
]

# --- Toplam = Ödenecek tutar (Amount payable) öncelik ---
_TOTAL_PAYABLE_PATTERNS: List[re.Pattern] = [
    # Parantez içi oranları atla: hemen ardından gelen gerçek TL tutarı hedefle
    re.compile(
        r"(?:ö?denecek\s*tutar|amount\s*payable|balance\s*due)"
        r"(?:\s*\([^)]*\))?\D{0,55}(" + _NUM + r")",
        re.I,
    ),
    re.compile(
        r"vergiler\s*dahil\s*toplam(?:\s*tutar)?(?:\s*\([^)]*\))?\D{0,55}(" + _NUM + r")",
        re.I,
    ),
    re.compile(
        r"(?:total\s*including\s*vat|total\s*payable)(?:\s*\([^)]*\))?\D{0,45}(" + _NUM + r")",
        re.I,
    ),
]

_TOTAL_SECONDARY_PATTERNS: List[re.Pattern] = [
    re.compile(r"vergiler\s*dahil\D{0,40}(" + _NUM + r")", re.I),
    re.compile(r"genel\s*toplam\D{0,40}(" + _NUM + r")", re.I),
    re.compile(r"grand\s*total\D{0,35}(" + _NUM + r")", re.I),
]

# --- Brüt mal/hizmet (iskonto öncesi) & iskonto satırı ---
_GROSS_GOODS_PATTERNS: List[re.Pattern] = [
    re.compile(r"mal\s*hizmet\s*toplam\s*tutar[ıi]?\D{0,30}(" + _NUM + r")", re.I),
    re.compile(r"mal\s*hizmet\s*toplam\D{0,20}:\s*(" + _NUM + r")", re.I),
]

_ISKONTO_PATTERNS: List[re.Pattern] = [
    re.compile(r"toplam\s*i?skonto\D{0,30}(" + _NUM + r")", re.I),
    re.compile(r"toplam\s*indirim\D{0,30}(" + _NUM + r")", re.I),
    re.compile(r"(?:total\s*discount)\D{0,30}(" + _NUM + r")", re.I),
]

_VKN_RE = re.compile(r"\bVKN\s*[:\s]*(\d{10})\b", re.I)
_TCKN_RE = re.compile(r"\bTCKN\s*[:\s]*(\d{11})\b", re.I)


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
    if num_str is None:
        return None
    s = str(num_str).replace(" ", "").replace("TL", "").replace("tl", "")
    s = (
        s.replace("B", "8")
        .replace("O", "0")
        .replace("S", "5")
        .replace("I", "1")
        .replace("l", "1")
        .replace("G", "6")
    )
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


def _line_ok_for_money(tx: str, *, for_total: bool = False) -> bool:
    if _PRODUCT_ROW_HINT.search(tx):
        return False
    if for_total and _SKIP_FOR_TOTAL.search(_normalize_whitespace(tx).lower()):
        return False
    return True


def _extract_labeled_bottom_first(
    patterns: List[re.Pattern],
    lines_bottom_first: List[Dict[str, Any]],
    blob: str,
    *,
    for_total: bool = False,
    min_value: float = 0.0,
    line_filter: Optional[Callable[[str], bool]] = None,
    use_blob_fallback: bool = True,
) -> Optional[float]:
    """Etikete göre ilk eşleşme; alt satırlar önce (fatura özeti)."""
    for ln in lines_bottom_first:
        tx = ln["text"]
        if line_filter is not None and not line_filter(tx):
            continue
        if not _line_ok_for_money(tx, for_total=for_total):
            continue
        for pat in patterns:
            m = pat.search(tx)
            if m:
                v = parse_tr_number(m.group(1))
                if v is not None and v >= min_value and v < 50_000_000:
                    return float(v)
    if use_blob_fallback:
        for pat in patterns:
            m = pat.search(blob)
            if m:
                v = parse_tr_number(m.group(1))
                if v is not None and v >= min_value and v < 50_000_000:
                    return float(v)
    return None


def _money_positions(line: str) -> List[Tuple[int, float]]:
    """Satırdaki TL tutarları; parantez içi %20.00 gibi oranlar hariç."""
    masked = _RATE_IN_PARENS.sub(" ", line)
    out: List[Tuple[int, float]] = []
    for m in _MONEY_TOKEN_RE.finditer(masked):
        v = parse_tr_number(m.group(0))
        if v is not None and 0 <= v < 50_000_000:
            out.append((m.start(), float(v)))
    return out


# Sıra: önce Ödenecek Tutar (toplam sabiti), sonra diğer ödenecek / vergiler dahil etiketleri
_ODENECEK_TUTAR_LABEL = re.compile(r"ö?denecek\s*tutar", re.I)
_TOTAL_LINE_LABELS: List[re.Pattern] = [
    re.compile(r"ö?denecek\s*tutar", re.I),
    re.compile(r"amount\s*payable|balance\s*due", re.I),
    re.compile(r"vergiler\s*dahil\s*toplam(?:\s*tutar)?", re.I),
    re.compile(r"total\s*(?:including\s*)?vat|total\s*payable", re.I),
]

_NET_LINE_LABELS: List[re.Pattern] = [
    re.compile(r"kdv\s*matrah[ıi]?", re.I),
    re.compile(r"vergi\s*hari[cç]\s*tutar", re.I),
    re.compile(r"vergi\s*matrah", re.I),
    re.compile(r"(?:vat\s*base|tax\s*base|taxable\s*amount)", re.I),
    re.compile(r"\bvergi\s*hari[cç]\b(?!\s*matrah)", re.I),
]


def _line_is_kdv_amount_line(tx: str) -> bool:
    if re.search(r"hesaplanan\s*kdv", tx, re.I):
        return True
    if re.search(r"calculated\s*vat|vat\s*amount", tx, re.I):
        return True
    if re.search(r"kdv\s*tutar[ıi]?", tx, re.I) and "matrah" not in tx.lower():
        return True
    return False


def _line_is_kdv_candidate(tx: str) -> bool:
    if not _line_is_kdv_amount_line(tx):
        return False
    if re.search(r"ö?denecek\s*tutar", tx, re.I) and not re.search(
        r"hesaplanan\s*kdv", tx, re.I
    ):
        return False
    if re.search(r"vergiler\s*dahil\s*toplam", tx, re.I) and not re.search(
        r"hesaplanan\s*kdv", tx, re.I
    ):
        return False
    return True


def _max_total_from_labeled_summary_lines(
    lines_bottom_first: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Özet bölümünde birden fazla 'toplam' tutarı varsa (ara toplam vs ödenecek),
    ödenecek / vergiler dahil / genel toplam etiketli satırlardaki TL'lerden en büyüğünü al.
    Örn. 485,81 ödenecek ile 80,97 KDV karışmasını önlemek için.
    """
    vals: List[float] = []
    for ln in lines_bottom_first[:50]:
        tx = ln["text"]
        if _is_discount_only_line(tx):
            continue
        if not _line_ok_for_money(tx, for_total=True):
            continue
        labeled = any(lab.search(tx) for lab in _TOTAL_LINE_LABELS) or bool(
            re.search(r"\bgenel\s*toplam\b|\bgrand\s*total\b", tx, re.I)
        )
        if not labeled:
            continue
        for _, v in _money_positions(tx):
            if 0.01 <= v < 50_000_000:
                vals.append(float(v))
    return max(vals) if vals else None


def _extract_payable_total_line_based(
    lines_bottom_first: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Toplam: öncelikle 'Ödenecek Tutar' satırındaki en büyük TL;
    yoksa diğer ödenecek / vergiler dahil toplam etiketleri.
    """
    for ln in lines_bottom_first:
        tx = ln["text"]
        if _is_discount_only_line(tx):
            continue
        if not _line_ok_for_money(tx, for_total=True):
            continue
        if _ODENECEK_TUTAR_LABEL.search(tx):
            nums = _money_positions(tx)
            if nums:
                return max(v for _, v in nums)
    for ln in lines_bottom_first:
        tx = ln["text"]
        if _is_discount_only_line(tx):
            continue
        if not _line_ok_for_money(tx, for_total=True):
            continue
        if _ODENECEK_TUTAR_LABEL.search(tx):
            continue
        for lab in _TOTAL_LINE_LABELS:
            if lab.search(tx):
                nums = _money_positions(tx)
                if not nums:
                    continue
                return max(v for _, v in nums)
    return None


def _extract_net_line_based(lines_bottom_first: List[Dict[str, Any]]) -> Optional[float]:
    """KDV matrahı / vergi hariç tutar — etiketten sonraki ilk anlamlı TL tutarı."""
    for ln in lines_bottom_first:
        tx = ln["text"]
        if _is_discount_only_line(tx):
            continue
        if not _line_ok_for_money(tx, for_total=False):
            continue
        if _PRODUCT_ROW_HINT.search(tx):
            continue
        for lab in _NET_LINE_LABELS:
            mm = lab.search(tx)
            if not mm:
                continue
            nums = _money_positions(tx)
            if not nums:
                continue
            start = mm.end()
            after = [(p, v) for p, v in nums if p >= start - 2]
            if after:
                return float(after[0][1])
            return float(nums[-1][1])
    return None


def _kdv_amount_from_line_after_label(tx: str, sub_start: int) -> Optional[float]:
    fragment = tx[sub_start:]
    masked = _RATE_IN_PARENS.sub(" ", fragment)
    for mm in _MONEY_TOKEN_RE.finditer(masked):
        v = parse_tr_number(mm.group(0))
        if v is not None and 0 <= v < 50_000_000:
            return float(v)
    return None


def _extract_kdv_line_based(lines_bottom_first: List[Dict[str, Any]]) -> Optional[float]:
    """
    Önce 'Hesaplanan KDV' satırı (yanındaki TL, örn. 80,97); yoksa KDV Tutarı / calculated VAT.
    Parantez içi %20.00 oran olarak alınmaz.
    """
    # 1) Hesaplanan KDV — öncelik
    for ln in lines_bottom_first:
        tx = ln["text"]
        if not _line_ok_for_money(tx, for_total=False):
            continue
        if not _line_is_kdv_candidate(tx):
            continue
        m = re.search(r"hesaplanan\s*kdv", tx, re.I)
        if not m:
            continue
        v = _kdv_amount_from_line_after_label(tx, m.end())
        if v is not None:
            return v
    # 2) KDV Tutarı / calculated VAT
    for ln in lines_bottom_first:
        tx = ln["text"]
        if not _line_ok_for_money(tx, for_total=False):
            continue
        if not _line_is_kdv_candidate(tx):
            continue
        if re.search(r"hesaplanan\s*kdv", tx, re.I):
            continue
        m2 = re.search(r"kdv\s*tutar[ıi]?", tx, re.I)
        if m2:
            sub_start = m2.end()
        else:
            m3 = re.search(r"(?:calculated\s*vat|vat\s*amount)", tx, re.I)
            if not m3:
                continue
            sub_start = m3.end()
        v = _kdv_amount_from_line_after_label(tx, sub_start)
        if v is not None:
            return v
    return None


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


def _lines_bottom_first(lines_y: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(lines_y, key=lambda x: x["y"], reverse=True)


def infer_tax_id_from_text(text: str) -> str:
    m = _VKN_RE.search(text or "")
    if m:
        return m.group(1).strip()
    m = _TCKN_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return ""


def infer_vendor_fallback(ocr_text: str) -> str:
    lines = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()]
    for ln in lines[:25]:
        low = ln.lower()
        if any(
            x in low
            for x in ("limited", "ltd", "a.ş", "a.s", "şti", "sti", "sanayi", "ticaret", "anonim")
        ):
            if 4 < len(ln) < 130:
                return ln[:128]
    for ln in lines[:8]:
        if 8 <= len(ln) <= 100 and not re.match(r"^[\d\s.,:/\-TLtl]+$", ln):
            return ln[:128]
    return lines[0][:80] if lines else ""


def _resolve_tesseract_executable() -> Optional[str]:
    env_p = (os.environ.get("TESSERACT_CMD") or "").strip()
    if env_p and os.path.isfile(env_p):
        return env_p
    w = shutil.which("tesseract")
    if w and os.path.isfile(w):
        return w
    for p in ("/usr/bin/tesseract", "/usr/local/bin/tesseract"):
        if os.path.isfile(p):
            return p
    return None


def _configure_tesseract_cmd() -> bool:
    path = _resolve_tesseract_executable()
    if path:
        pytesseract.pytesseract.tesseract_cmd = path
        return True
    return False


def _ocr_lines_from_tesseract_dict(data: Dict[str, Any]) -> List[OcrLine]:
    n = len(data.get("text", []))
    line_words: Dict[Tuple[int, int, int], List[Tuple[int, str, float, float]]] = defaultdict(list)
    for i in range(n):
        raw_t = data["text"][i] or ""
        t = raw_t.strip()
        if not t:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0:
            conf = 0.0
        block = int(data["block_num"][i])
        par = int(data["par_num"][i])
        line = int(data["line_num"][i])
        left = int(data["left"][i])
        top = int(data["top"][i])
        height = int(data["height"][i])
        y_c = float(top) + float(height) / 2.0
        line_words[(block, par, line)].append((left, t, conf, y_c))
    lines: List[OcrLine] = []
    for key in sorted(line_words.keys()):
        parts = sorted(line_words[key], key=lambda x: x[0])
        joined = _normalize_whitespace(" ".join(p[1] for p in parts))
        if not joined:
            continue
        ys = [p[3] for p in parts]
        y_center = sum(ys) / len(ys) if ys else 0.0
        confs = [p[2] for p in parts if p[2] > 0]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        lines.append(OcrLine(text=joined, confidence=float(avg_conf), y_center=y_center))
    lines.sort(key=lambda ln: ln.y_center)
    return lines


def _run_ocr_tesseract_core(gray: Any) -> List[OcrLine]:
    cfg = "--oem 3 --psm 6"
    try:
        data = pytesseract.image_to_data(
            gray,
            lang="tur+eng",
            config=cfg,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError:
        data = pytesseract.image_to_data(
            gray,
            lang="eng",
            config=cfg,
            output_type=pytesseract.Output.DICT,
        )
    return _ocr_lines_from_tesseract_dict(data)


@st.cache_resource(show_spinner=False)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["tr", "en"], gpu=False)


def _ocr_text_usable(text: str) -> bool:
    """Yetersiz OCR çıktısında bir sonraki motor devreye girsin (sahte 'boş' sonuç yok)."""
    t = (text or "").strip()
    if len(t) < 18:
        return False
    digitish = sum(1 for c in t if c.isdigit())
    return digitish >= 3


def _run_ocr_gemini_vision(pil_img: Image.Image, api_key: str) -> Dict[str, Any]:
    """Gemini Vision — Tesseract/EasyOCR yetersiz kaldığında tam metin çıkarımı."""
    try:
        import google.generativeai as genai
    except ImportError:
        return {"text": "", "lines": [], "ocr_engine": None}
    key = (api_key or "").strip()
    if not key:
        return {"text": "", "lines": [], "ocr_engine": None}
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    img = pil_img.convert("RGB")
    prompt = (
        "You are a precise OCR engine for Turkish invoices. Transcribe EVERY visible character "
        "in reading order. Preserve line breaks. Keep number punctuation exactly as printed "
        "(e.g. 80,97 TL, 763,00, 1.234,56). "
        "Output ONLY the raw transcript, no titles or markdown."
    )
    try:
        resp = model.generate_content(
            [img, prompt],
            generation_config={"temperature": 0.05, "max_output_tokens": 8192},
        )
    except Exception:
        logger.exception("Gemini Vision request failed")
        return {"text": "", "lines": [], "ocr_engine": None}
    text = ""
    try:
        text = (resp.text or "").strip()
    except Exception:
        cand = getattr(resp, "candidates", None) or []
        if cand:
            parts = getattr(cand[0].content, "parts", []) or []
            for p in parts:
                if getattr(p, "text", None):
                    text += p.text
        text = text.strip()
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return {"text": "", "lines": [], "ocr_engine": "gemini_vision"}
    lines_y = _lines_from_text(text)
    ocr_lines = [
        {"text": ln["text"], "y_center": float(ln["y"]), "confidence": 0.88} for ln in lines_y
    ]
    return {
        "text": "\n".join(ln["text"] for ln in lines_y),
        "lines": ocr_lines,
        "ocr_engine": "gemini_vision",
    }


def _run_ocr_easyocr(pil_img: Image.Image) -> Dict[str, Any]:
    reader = _easyocr_reader()
    img_np = np.asarray(pil_img.convert("RGB"))
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
        "ocr_engine": "easyocr",
    }


def run_ocr(
    pil_img: Image.Image,
    *,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Hibrit OCR: Tesseract → (hata veya zayıf metin) EasyOCR → (hâlâ zayıf + API varsa) Gemini Vision.
    """
    img_np = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    best: Dict[str, Any] = {"text": "", "lines": [], "ocr_engine": None}

    if _configure_tesseract_cmd():
        try:
            lines = _run_ocr_tesseract_core(gray)
            full_text = "\n".join(ln.text for ln in lines)
            tess_out = {
                "text": full_text,
                "lines": [ln.__dict__ for ln in lines],
                "ocr_engine": "tesseract",
            }
            if _ocr_text_usable(full_text):
                return tess_out
            best = tess_out
        except Exception:
            logger.exception("Tesseract OCR failed")

    try:
        eo = _run_ocr_easyocr(pil_img)
        t = str(eo.get("text") or "")
        if _ocr_text_usable(t):
            return eo
        if len(t.strip()) > len(str(best.get("text") or "").strip()):
            best = eo
    except Exception:
        logger.exception("EasyOCR failed")

    gkey = (gemini_api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if gkey and gemini_is_configured(gkey):
        try:
            gv = _run_ocr_gemini_vision(pil_img, gkey)
            gt = str(gv.get("text") or "")
            if _ocr_text_usable(gt):
                return gv
            if len(gt.strip()) > len(str(best.get("text") or "").strip()):
                best = gv
        except Exception:
            logger.exception("Gemini Vision OCR failed")

    return best


def extract_invoice_fields(
    text: str, ocr_lines: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Toplam: 'Ödenecek Tutar' satırı; KDV: 'Hesaplanan KDV' satırı; Net: KDV matrahı.
    Net + KDV ≈ toplam: satırdan okunan KDV bu üçlüyle uyumluysa korunur; aksi halde
    KDV = toplam − net ile tamamlanır (matrah hatasında satır KDV'si ezilmez).
    """
    lines_y = _parse_ocr_lines(ocr_lines)
    if not lines_y:
        lines_y = _lines_from_text(text)
    blob = _normalize_whitespace(" ".join(ln["text"] for ln in lines_y))
    bottom_first = _lines_bottom_first(lines_y)

    odenecek = _extract_payable_total_line_based(bottom_first)
    if odenecek is None:
        odenecek = _extract_labeled_bottom_first(
            _TOTAL_PAYABLE_PATTERNS, bottom_first, blob, for_total=True
        )
    if odenecek is None:
        odenecek = _extract_labeled_bottom_first(
            _TOTAL_SECONDARY_PATTERNS, bottom_first, blob, for_total=True
        )

    net = _extract_net_line_based(bottom_first)
    if net is None:
        net = _extract_labeled_bottom_first(_NET_STRICT_PATTERNS, bottom_first, blob)
    if net is None:
        net = _extract_labeled_bottom_first(_NET_FALLBACK_PATTERNS, bottom_first, blob)

    kdv = _extract_kdv_line_based(bottom_first)
    if kdv is None:
        kdv = _extract_labeled_bottom_first(
            _KDV_PATTERNS,
            bottom_first,
            blob,
            min_value=0.0,
            line_filter=_line_is_kdv_candidate,
            use_blob_fallback=False,
        )

    brut_mal = _extract_labeled_bottom_first(_GROSS_GOODS_PATTERNS, bottom_first, blob)
    iskonto_satir = _extract_labeled_bottom_first(_ISKONTO_PATTERNS, bottom_first, blob)

    tahmini_iskonto: Optional[float] = None
    if iskonto_satir is not None:
        tahmini_iskonto = float(iskonto_satir)
    elif brut_mal is not None and odenecek is not None:
        diff = float(_to_decimal_2(brut_mal) - _to_decimal_2(odenecek))
        if diff > GOLDEN_RULE_TOL:
            tahmini_iskonto = diff

    genel_toplam = odenecek
    max_labeled = _max_total_from_labeled_summary_lines(bottom_first)
    if max_labeled is not None:
        if genel_toplam is None:
            genel_toplam = max_labeled
        else:
            g0 = float(_to_decimal_2(genel_toplam))
            if max_labeled > g0 + 0.02:
                genel_toplam = max_labeled

    if genel_toplam is None and net is not None and kdv is not None:
        genel_toplam = float(_to_decimal_2(net) + _to_decimal_2(kdv))

    # Altın kural: net + KDV ≈ toplam. Satırdan okunan "Hesaplanan KDV" tutarlıysa EZME;
    # her zaman kdv = toplam − net yapmak, matrah hatasında KDV'yi de bozar.
    if genel_toplam is not None and net is not None:
        g = _to_decimal_2(genel_toplam)
        n = _to_decimal_2(net)
        if n <= g + GOLDEN_RULE_TOL:
            implied_kdv = float(max(Decimal("0"), g - n))
            if kdv is None:
                kdv = implied_kdv
            else:
                sk = _to_decimal_2(kdv)
                if abs((n + sk) - g) <= _KDV_LINE_SUM_TOL:
                    kdv = float(sk)
                elif abs(sk - _to_decimal_2(implied_kdv)) <= Decimal("0.05"):
                    kdv = float(sk)
                else:
                    kdv = implied_kdv
    elif genel_toplam is not None and kdv is not None and net is None:
        d = float(_to_decimal_2(genel_toplam) - _to_decimal_2(kdv))
        if d > 0:
            net = d
    elif net is not None and kdv is not None and genel_toplam is None:
        genel_toplam = float(_to_decimal_2(net) + _to_decimal_2(kdv))

    return {
        "firma_adi": infer_vendor_fallback(text or ""),
        "vergi_no": infer_tax_id_from_text(text or "") or None,
        "tarih": datetime.now().strftime("%Y-%m-%d"),
        "net_tutar": net,
        "kdv_tutari": kdv,
        "genel_toplam": genel_toplam,
        "brut_mal_hizmet": brut_mal,
        "tahmini_iskonto": tahmini_iskonto,
    }


def validate_golden_rule(
    net_tutar: Optional[float],
    kdv_tutari: Optional[float],
    genel_toplam: Optional[float],
) -> Dict[str, Any]:
    if net_tutar is None or kdv_tutari is None or genel_toplam is None:
        return {
            "ok": None,
            "label": "Analiz ediliyor…",
            "box": "gray",
            "detail": "Tutarlar henüz okunamadı veya eksik.",
            "message": "Analiz ediliyor…",
        }
    n = _to_decimal_2(net_tutar)
    k = _to_decimal_2(kdv_tutari)
    t = _to_decimal_2(genel_toplam)
    diff = abs((n + k) - t)
    # Büyük tutarlarda OCR/yuvarlama: küçük göreli sapma (üst sınır ~5 TL)
    rel_extra = (t.copy_abs() * Decimal("0.0001")).quantize(Decimal("0.01"))
    if rel_extra < Decimal("0.01"):
        rel_extra = Decimal("0.01")
    if rel_extra > Decimal("5.00"):
        rel_extra = Decimal("5.00")
    tol = GOLDEN_RULE_VALIDATION_TOL + rel_extra
    if diff <= tol:
        return {
            "ok": True,
            "label": "GÜVENLİ",
            "box": "green",
            "detail": "Matematiksel doğrulama başarılı: Net + KDV, toplam ile uyumlu.",
            "message": "GÜVENLİ: Net + KDV ile toplam tutarlı.",
        }
    diff_f = float(diff)
    return {
        "ok": False,
        "label": "RİSKLİ",
        "box": "red",
        "detail": (
            f"Net + KDV − Toplam = {diff_f:,.2f} TL (tolerans ±{float(tol):,.2f} TL dışında). "
            "Tutarları fişte kontrol edin veya aşağıdan düzenleyin."
        ).replace(",", "X").replace(".", ",").replace("X", "."),
        "message": (
            f"RİSKLİ: Fark {diff_f:,.2f} TL — net+KDV ile ödenecek tutar örtüşmüyor."
        )
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", "."),
    }


def gemini_is_configured(api_key: Optional[str] = None) -> bool:
    """Anahtar var mı (çağıran, Streamlit secrets dahil, dolu string geçmeli). Ortam: GEMINI_API_KEY."""
    return bool((api_key or os.environ.get("GEMINI_API_KEY") or "").strip())


def gemini_auditor_note(
    ocr_text: str,
    fields: Dict[str, Any],
    api_key: Optional[str],
    *,
    model_name: str = "gemini-1.5-flash",
) -> str:
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return ""
    try:
        import google.generativeai as genai
    except ImportError:
        return ""
    payload = {
        "net": fields.get("net_tutar"),
        "kdv": fields.get("kdv_tutari"),
        "total_payable": fields.get("genel_toplam"),
        "discount_hint": fields.get("tahmini_iskonto"),
    }
    prompt = (
        "You are a senior financial auditor. OCR excerpt (may be noisy):\n"
        f"\"\"\"{(ocr_text or '')[:6000]}\"\"\"\n\n"
        f"Extracted amounts (TL): {json.dumps(payload, ensure_ascii=False)}\n\n"
        "Write exactly two complete sentences in English. Sentence 1: identify the likely "
        "invoice type or sector (e.g. cosmetics, IT services) and any notable line items. "
        "Sentence 2: state whether VAT/totals look consistent with the extracted net + VAT ≈ total "
        "and mention one brief compliance or risk note. "
        "Tone: formal, neutral. No markdown, bullets, or title."
    )
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 220},
        )
        out = (resp.text or "").strip()
        return out[:600] if out else ""
    except Exception:
        return ""


def infer_invoice_category_heuristic(ocr_text: str) -> Dict[str, str]:
    """API yokken veya yedek: OCR metninden kaba kategori tahmini."""
    t = _normalize_whitespace(ocr_text or "").lower()
    rules: List[Tuple[str, str, List[str]]] = [
        ("Yemek / Restoran", "Food & beverage", ["yemek", "restoran", "lokanta", "gıda", "cafe", "café", "menu", "kebap"]),
        ("Akaryakıt", "Fuel", ["akaryakıt", "petrol", "shell", "opet", "bp ", "istasyon", "mazot"]),
        ("Kozmetik & Kişisel bakım", "Cosmetics", ["kozmetik", "parfüm", "bakım", "krem", "şampuan", "makyaj"]),
        ("Kırtasiye / Ofis", "Office supplies", ["kırtasiye", "ofis", "kağıt", "toner", "kalem"]),
        ("Konaklama / Seyahat", "Travel", ["otel", "konaklama", "seyahat", "pansiyon", "uçak"]),
        ("IT / Yazılım", "IT / Software", ["yazılım", "lisans", "hosting", "domain", "microsoft", "google"]),
        ("Sağlık", "Healthcare", ["eczane", "ilaç", "hastane", "sağlık", "medikal"]),
        ("Perakende / Market", "Retail", ["market", "migros", "a101", "bim ", "şok ", "carrefour"]),
    ]
    for tr, en, keys in rules:
        if any(k in t for k in keys):
            return {"category_tr": tr, "category_en": en, "source": "heuristic"}
    return {"category_tr": "Genel / Belirsiz", "category_en": "General / Unclassified", "source": "heuristic"}


def gemini_invoice_enrichment(
    ocr_text: str,
    fields: Dict[str, Any],
    api_key: Optional[str],
    *,
    model_name: str = "gemini-1.5-flash",
) -> Dict[str, Any]:
    """
    Fiş/fatura kategorisi + 2 cümlelik denetçi notu (tek Gemini çağrısı).
    Anahtar yoksa yalnızca anahtar kelime sezgisel kategori döner.
    """
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        h = infer_invoice_category_heuristic(ocr_text)
        return {
            "category_tr": h["category_tr"],
            "category_en": h["category_en"],
            "auditor_note": "",
            "source": h.get("source", "heuristic"),
        }
    try:
        import google.generativeai as genai
    except ImportError:
        h = infer_invoice_category_heuristic(ocr_text)
        return {
            "category_tr": h["category_tr"],
            "category_en": h["category_en"],
            "auditor_note": "",
            "source": "heuristic",
        }

    payload = {
        "net": fields.get("net_tutar"),
        "kdv": fields.get("kdv_tutari"),
        "total": fields.get("genel_toplam"),
        "firma": fields.get("firma_adi"),
    }
    prompt = (
        "Sen Türkçe fatura ve fiş metinlerini sınıflandıran bir muhasebe asistanısın. "
        "Yanıtın YALNIZCA geçerli JSON olsun (markdown, kod çiti, açıklama yok).\n"
        'Şema: {"category_tr": string, "category_en": string, "auditor_note": string}\n'
        "- category_tr: gider türü (kısa Türkçe, örn. Kozmetik, Yemek, Akaryakıt, IT, Genel).\n"
        "- category_en: aynı kategori İngilizce.\n"
        "- auditor_note: tam iki cümle Türkçe — (1) belgenin muhtemel içeriği/sektörü, "
        "(2) Net+KDV ile genel toplamın tutarlılığı ve kısa bir uyum/ risk notu.\n\n"
        f"Çıkarılmış tutarlar: {json.dumps(payload, ensure_ascii=False)}\n\n"
        f"OCR metni:\n{(ocr_text or '')[:7500]}"
    )
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 500},
        )
        raw = (resp.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I | re.MULTILINE)
        raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                "category_tr": str(data.get("category_tr") or "Genel")[:160],
                "category_en": str(data.get("category_en") or "General")[:160],
                "auditor_note": str(data.get("auditor_note") or "")[:1200],
                "source": "gemini",
            }
    except Exception:
        logger.exception("gemini_invoice_enrichment failed")

    h = infer_invoice_category_heuristic(ocr_text)
    return {
        "category_tr": h["category_tr"],
        "category_en": h["category_en"],
        "auditor_note": "",
        "source": "heuristic_fallback",
    }


def init_sustainability_counter(*, grams_per_invoice: float = 0.02) -> None:
    if "sustainability_paper_saved_g" not in st.session_state:
        st.session_state["sustainability_paper_saved_g"] = 0.0
    if "sustainability_grams_per_invoice" not in st.session_state:
        st.session_state["sustainability_grams_per_invoice"] = float(grams_per_invoice)


def increment_sustainability_counter(increment_by: int = 1) -> None:
    init_sustainability_counter()
    inc = max(int(increment_by or 0), 0)
    gpi = float(st.session_state.get("sustainability_grams_per_invoice", 0.02))
    prev = float(st.session_state.get("sustainability_paper_saved_g", 0.0))
    st.session_state["sustainability_paper_saved_g"] = float(
        _to_decimal_2(Decimal(str(prev)) + Decimal(str(inc * gpi)))
    )


def get_paper_saved_grams() -> float:
    init_sustainability_counter()
    return float(st.session_state.get("sustainability_paper_saved_g", 0.0))
