# Audit-Sight/utils.py
"""
AuditSight Pro — OCR, sustainability stats, VAT validation (golden-rule cross-check).
"""
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

# Legacy micro-tolerance for rate heuristics
TOL = Decimal("0.05")
# Golden rule: Grand Total valid only if Net + KDV matches within this (spec: < 0.10)
GOLDEN_RULE_TOL = Decimal("0.10")
# Auditor override threshold: if |Calculated - Potential| > 1.0, force Calculated.
TOTAL_OVERRIDE_TOL = Decimal("1.00")

_DISCOUNT_LINE = re.compile(
    r"iskonto|indirim|isk\.|iade|discount|tevkifat",
    re.IGNORECASE,
)

# Grand total: keyword → priority (higher = stronger anchor)
_TOTAL_KEYWORD_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"ödenecek\s*tutar\D{0,50}(\d+(?:[.,]\d{2})?)", re.I), 100),
    (re.compile(r"vergiler\s*dahil\D{0,50}(\d+(?:[.,]\d{2})?)", re.I), 98),
    (re.compile(r"genel\s*toplam\D{0,50}(\d+(?:[.,]\d{2})?)", re.I), 95),
    (re.compile(r"toplam\s*tutar\D{0,50}(\d+(?:[.,]\d{2})?)", re.I), 92),
    (re.compile(r"(?<![\w])toplam\D{0,25}(\d+(?:[.,]\d{2})?)", re.I), 70),
]

_NET_PATTERNS = [
    re.compile(r"(?:matrah|vergi\s*matrahı|kdv\s*matrahı|net\s*tutar)\D{0,45}(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"(?:ara\s*toplam|mal\s*hizmet|tutar)(?:\s*tutarı)?\D{0,35}(\d+(?:[.,]\d{2})?)", re.I),
]

_KDV_PATTERNS = [
    re.compile(r"(?:hesaplanan\s*)?kdv\D{0,40}(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"k\.?\s*d\.?\s*v\.?\D{0,35}(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"(?:katma\s*değer|vergi\s*tutar(?:ı|i))\D{0,35}(\d+(?:[.,]\d{2})?)", re.I),
]

_KDV_20_PATTERNS = [
    re.compile(r"(?:%|oran[:\s]*)\s*20\D{0,40}(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"kdv\s*%?\s*20\D{0,40}(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"%\s*20\D{0,20}kdv\D{0,40}(\d+(?:[.,]\d{2})?)", re.I),
]

_MONEY_IN_LINE = re.compile(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2}")
_KDV_RATE_HINT = re.compile(r"%\s*(20|10|1)\b|kdv\s*%?\s*(20|10|1)\b", re.I)
_BOTTOM_TOTAL_HINT = re.compile(r"ödenecek\s*tutar|genel\s*toplam|toplam", re.I)


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


def _is_ara_toplam_line(text: str) -> bool:
    """Avoid treating mid-invoice 'Ara Toplam' as grand total."""
    return bool(re.search(r"ara\s*toplam|mal\s*hizmet\s*toplam", _safe_lower(text or ""), re.I))


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
    lines_y_asc: List[Dict[str, Any]], patterns: List[re.Pattern], *, allow_zero: bool = False
) -> List[Tuple[float, float]]:
    found: List[Tuple[float, float]] = []
    for ln in lines_y_asc:
        if _is_discount_line(ln["text"]):
            continue
        for pat in patterns:
            m = pat.search(ln["text"])
            if not m:
                continue
            v = parse_tr_number(m.group(1))
            valid = v is not None and (0 <= v < 50_000_000 if allow_zero else 0 < v < 50_000_000)
            if valid:
                found.append((float(v), float(ln["y"])))
                break
    return found


def _grand_total_candidates(lines_bottom_first: List[Dict[str, Any]]) -> List[Tuple[float, int, float]]:
    """
    Returns list of (amount, keyword_priority, y) — scan bottom → top.
    """
    cands: List[Tuple[float, int, float]] = []
    seen: set = set()
    n_lines = len(lines_bottom_first)

    for idx, ln in enumerate(lines_bottom_first):
        if _is_discount_line(ln["text"]):
            continue
        text = ln["text"]
        y = float(ln["y"])
        # Do not anchor grand total on subtotal rows (e.g. "Ara Toplam Tutar" matching "toplam tutar")
        skip_subtotal_row = _is_ara_toplam_line(text)
        for pat, pri in _TOTAL_KEYWORD_PATTERNS:
            if skip_subtotal_row:
                continue
            m = pat.search(text)
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

    if n_lines:
        cutoff = max(0, n_lines * 2 // 3)
        for ln in lines_bottom_first[: max(15, n_lines - cutoff)]:
            if _is_discount_line(ln["text"]) or _is_ara_toplam_line(ln["text"]):
                continue
            for m in _MONEY_IN_LINE.finditer(ln["text"]):
                v = parse_tr_number(m.group(0))
                if v is None or v <= 0 or v > 50_000_000:
                    continue
                key = (round(v, 2), 25)
                if key in seen:
                    continue
                seen.add(key)
                cands.append((float(v), 25, float(ln["y"])))

    cands.sort(key=lambda t: (-t[1], -t[2]))
    return cands


def _pick_net_kdv_for_math(
    net_hits: List[Tuple[float, float]],
    kdv_hits: List[Tuple[float, float]],
    max_each: int = 6,
) -> List[Tuple[float, float]]:
    if not net_hits or not kdv_hits:
        return []
    nets = sorted({round(a[0], 2): a for a in net_hits}.values(), key=lambda x: -x[1])[:max_each]
    kdvs = sorted({round(a[0], 2): a for a in kdv_hits}.values(), key=lambda x: -x[1])[:max_each]
    pairs: List[Tuple[float, float]] = []
    for nv, _ in nets:
        for kv, _ in kdvs:
            pairs.append((nv, kv))
    return pairs


def _extract_last_currency_value(lines_bottom_first: List[Dict[str, Any]]) -> Optional[float]:
    for ln in lines_bottom_first:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        matches = list(_MONEY_IN_LINE.finditer(tx))
        if not matches:
            continue
        for m in reversed(matches):
            v = parse_tr_number(m.group(0))
            if v is not None and 0 < v < 50_000_000:
                return float(v)
    return None


def _collect_all_money_amounts(lines_y_asc: List[Dict[str, Any]]) -> List[float]:
    vals: List[float] = []
    for ln in lines_y_asc:
        for m in _MONEY_IN_LINE.finditer(ln["text"]):
            v = parse_tr_number(m.group(0))
            if v is not None and 0 <= v < 50_000_000:
                vals.append(float(v))
    return vals


def _is_vat_rate_like_amount(vat_amount: float) -> bool:
    if vat_amount is None:
        return False
    if vat_amount < 0 or vat_amount > 30:
        return False
    common_rates = {1.0, 8.0, 10.0, 18.0, 20.0}
    return round(vat_amount, 2) in common_rates


def _resolve_vat_amount_shield(
    net_amount: Optional[float],
    extracted_vat: Optional[float],
    all_money_values: List[float],
) -> Optional[float]:
    if net_amount is None or extracted_vat is None:
        return extracted_vat
    if net_amount <= 0 or not _is_vat_rate_like_amount(extracted_vat):
        return extracted_vat

    rate_decimal = _to_decimal_2(extracted_vat) / Decimal("100")
    expected_vat = float(_to_decimal_2(_to_decimal_2(net_amount) * rate_decimal))
    best_match: Optional[float] = None
    best_diff = Decimal("999999")
    for raw in all_money_values:
        candidate = _to_decimal_2(raw)
        if abs(candidate - _to_decimal_2(extracted_vat)) <= Decimal("0.05"):
            continue
        diff = abs(candidate - _to_decimal_2(expected_vat))
        if diff <= Decimal("0.25") and diff < best_diff:
            best_diff = diff
            best_match = float(candidate)
    return best_match if best_match is not None else extracted_vat


def _bottom_number_candidates(lines_bottom_first: List[Dict[str, Any]], max_rows: int = 20) -> List[float]:
    out: List[float] = []
    seen: set = set()
    for ln in lines_bottom_first[:max_rows]:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        for m in _MONEY_IN_LINE.finditer(tx):
            v = parse_tr_number(m.group(0))
            if v is None or v <= 0 or v > 50_000_000:
                continue
            key = round(v, 2)
            if key in seen:
                continue
            seen.add(key)
            out.append(float(v))
    return out


def _vat_candidates_from_lines(lines_y_asc: List[Dict[str, Any]]) -> List[float]:
    vals: List[Tuple[float, float]] = []
    for ln in lines_y_asc:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        has_kdv_hint = bool(_KDV_RATE_HINT.search(tx)) or "kdv" in _safe_lower(tx) or "vergi" in _safe_lower(tx)
        if not has_kdv_hint:
            continue
        for m in _MONEY_IN_LINE.finditer(tx):
            v = parse_tr_number(m.group(0))
            if v is None or v < 0 or v > 50_000_000:
                continue
            vals.append((float(v), float(ln["y"])))
    uniq = sorted({round(v, 2): (v, y) for v, y in vals}.values(), key=lambda t: -t[1])
    return [float(v) for v, _y in uniq]


def _solve_trio_net_vat_total(
    bottom_numbers: List[float],
    vat_candidates: List[float],
    tol: Decimal = GOLDEN_RULE_TOL,
) -> Optional[Tuple[float, float, float]]:
    if len(bottom_numbers) < 3:
        return None
    nums = sorted({round(v, 2): float(round(v, 2)) for v in bottom_numbers}.values())
    vat_set = {round(v, 2) for v in vat_candidates}
    best: Optional[Tuple[int, float, float, float]] = None
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if j <= i:
                continue
            c = float(_to_decimal_2(a) + _to_decimal_2(b))
            for total in nums:
                if total < max(a, b):
                    continue
                if abs(_to_decimal_2(total) - _to_decimal_2(c)) > tol:
                    continue
                if total + 1e-9 < max(a, b):
                    continue
                net, vat = (a, b) if a >= b else (b, a)
                if round(a, 2) in vat_set and round(b, 2) not in vat_set:
                    vat, net = a, b
                elif round(b, 2) in vat_set and round(a, 2) not in vat_set:
                    vat, net = b, a
                elif round(a, 2) in vat_set and round(b, 2) in vat_set:
                    vat, net = (a, b) if a <= b else (b, a)

                score = 0
                if round(vat, 2) in vat_set:
                    score += 100
                if round(total, 2) == round(max(a, b, total), 2):
                    score += 10
                if best is None or score > best[0] or (score == best[0] and total > best[3]):
                    best = (score, float(net), float(vat), float(total))
    if best is None:
        return None
    return best[1], best[2], best[3]


def _collect_currency_mentions(lines_y_asc: List[Dict[str, Any]]) -> List[Tuple[float, float, bool, bool]]:
    mentions: List[Tuple[float, float, bool, bool]] = []
    for ln in lines_y_asc:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        has_total_hint = bool(_BOTTOM_TOTAL_HINT.search(tx))
        has_vat_hint = bool(_KDV_RATE_HINT.search(tx)) or "kdv" in _safe_lower(tx) or "vergi" in _safe_lower(tx)
        for m in _MONEY_IN_LINE.finditer(tx):
            v = parse_tr_number(m.group(0))
            if v is None or v <= 0 or v > 50_000_000:
                continue
            mentions.append((float(v), float(ln["y"]), has_total_hint, has_vat_hint))
    return mentions


def _strict_triple_match(
    lines_y_asc: List[Dict[str, Any]],
    tol: Decimal = GOLDEN_RULE_TOL,
) -> Optional[Tuple[float, float, float]]:
    mentions = _collect_currency_mentions(lines_y_asc)
    if len(mentions) < 3:
        return None

    value_meta: Dict[float, Tuple[float, bool, bool]] = {}
    for v, y, total_hint, vat_hint in mentions:
        key = float(round(v, 2))
        prev = value_meta.get(key)
        if prev is None or y > prev[0]:
            value_meta[key] = (y, total_hint, vat_hint)
        else:
            value_meta[key] = (prev[0], prev[1] or total_hint, prev[2] or vat_hint)

    vals = sorted(value_meta.keys())
    if len(vals) < 3:
        return None

    anchor_ys = [y for _v, y, has_total, _has_vat in mentions if has_total]
    bottom_anchor = max(anchor_ys) if anchor_ys else max(y for _v, y, _t, _k in mentions)

    best: Optional[Tuple[Tuple[int, int, int], Tuple[float, float, float]]] = None
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            if j <= i:
                continue
            expected_total = float(_to_decimal_2(a) + _to_decimal_2(b))
            for c in vals:
                if abs(_to_decimal_2(c) - _to_decimal_2(expected_total)) > tol:
                    continue
                if c + 1e-9 < max(a, b):
                    continue

                a_meta = value_meta[a]
                b_meta = value_meta[b]
                vat = a if a <= b else b
                net = b if a <= b else a
                if a_meta[2] and not b_meta[2]:
                    vat, net = a, b
                elif b_meta[2] and not a_meta[2]:
                    vat, net = b, a

                vat_conf = 0
                if round(vat, 2) in {1.0, 10.0, 20.0}:
                    vat_conf -= 20
                if net > 0:
                    rate = _to_decimal_2(vat) / _to_decimal_2(net)
                    if abs(rate - Decimal("0.01")) <= Decimal("0.003"):
                        if abs(_to_decimal_2(net * 0.01) - _to_decimal_2(vat)) <= Decimal("0.10"):
                            vat_conf += 40
                        else:
                            vat_conf -= 40
                    elif abs(rate - Decimal("0.10")) <= Decimal("0.02") or abs(rate - Decimal("0.20")) <= Decimal("0.02"):
                        vat_conf += 20

                total_meta = value_meta[c]
                proximity = -int(abs(total_meta[0] - bottom_anchor) * 1000)
                if total_meta[1]:
                    proximity += 5000
                bottom_score = int(total_meta[0] * 1000)
                score = (proximity, vat_conf, bottom_score)

                trio = (float(net), float(vat), float(c))
                if best is None or score > best[0]:
                    best = (score, trio)

    if best is None:
        return None
    return best[1]


def _normalize_net_vat_assignment(
    net_value: float, vat_value: float
) -> Tuple[float, float]:
    """
    Logical Field Assignment:
    - Net should be larger than VAT in standard invoices.
    - If inverted, auto-swap.
    - Validate smaller/larger ratio against common VAT rates.
    """
    net_d = _to_decimal_2(net_value)
    vat_d = _to_decimal_2(vat_value)
    if vat_d > net_d:
        net_d, vat_d = vat_d, net_d

    if net_d <= Decimal("0.00"):
        return float(net_d), float(vat_d)

    ratio = vat_d / net_d if net_d > 0 else Decimal("0.00")
    standard_rates = (Decimal("0.20"), Decimal("0.10"), Decimal("0.01"))
    closest = min(standard_rates, key=lambda r: abs(ratio - r))

    # If still highly implausible after swap, keep magnitude order anyway.
    # This enforces "larger is Net, smaller is VAT" as requested.
    if abs(ratio - closest) > Decimal("0.25"):
        return float(max(net_d, vat_d)), float(min(net_d, vat_d))

    return float(net_d), float(vat_d)


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
            result.update(
                {
                    "status": "OK",
                    "label": f"✅ GÜVENLİ: Altın kural sağlandı; KDV oranı ~%{pct}",
                    "color": "green",
                }
            )
        elif sum_ok:
            result.update(
                {
                    "status": "ŞÜPHELİ",
                    "label": "⚠️ Net+KDV=Toplam (±0,10); KDV oranı standart değil.",
                    "color": "orange",
                }
            )
        else:
            diff = abs((net_d + kdv_eff) - total_d)
            result.update(
                {
                    "status": "RISK",
                    "label": f"🚨 Altın kural başarısız: Net+KDV ≠ Toplam (|fark|={float(diff):.2f} > 0,10).",
                    "color": "red",
                }
            )
    except Exception as exc:
        logger.exception("validate_vat: %s", exc)
        result.update(
            {
                "status": "ERROR",
                "label": "Denetim sırasında hata; verileri kontrol edin.",
                "color": "gray",
            }
        )
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
        ]
        for keywords, code in rules:
            if any(k in haystack for k in keywords):
                return code
        return "770 (Genel Gider)"
    except Exception:
        return "770 (Genel Gider)"


def run_ocr(pil_img: Image.Image) -> Dict[str, Any]:
    """
    OCR via system Tesseract only (tur+eng). Lines carry y_center for strict triple / math.
    """
    img_np = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _ocr_cfg = "--oem 3 --psm 6"
    try:
        data = pytesseract.image_to_data(
            gray,
            lang="tur+eng",
            config=_ocr_cfg,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError:
        # Eksik tur traineddata (ör. minimal apt kurulumu) durumunda en azından İngilizce ile devam et
        data = pytesseract.image_to_data(
            gray,
            lang="eng",
            config=_ocr_cfg,
            output_type=pytesseract.Output.DICT,
        )

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
        key = (block, par, line)
        line_words[key].append((left, t, conf, y_c))

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
    return {
        "text": "\n".join(ln.text for ln in lines),
        "lines": [ln.__dict__ for ln in lines],
    }


def extract_invoice_fields(
    text: str, ocr_lines: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Mathematical Truth Filter:
    Only assign financial fields when a valid Net + VAT = Total trio exists.
    """
    lines_y = _parse_ocr_lines(ocr_lines)
    if not lines_y:
        lines_y = _lines_from_text(text)

    all_money_values = _collect_all_money_amounts(lines_y)
    strict_trio = _strict_triple_match(lines_y)
    net = 0.0
    kdv = 0.0
    genel_toplam: Optional[float] = None
    total_source = "no_valid_math"
    override_applied = False

    def _apply_trio(
        trio: Tuple[float, float, float],
        *,
        source: str,
    ) -> None:
        nonlocal net, kdv, genel_toplam, total_source, override_applied
        n0, k0, g0 = trio
        net = n0
        kdv = float(_resolve_vat_amount_shield(net, k0, all_money_values) or 0.0)
        net, kdv = _normalize_net_vat_assignment(net, kdv)
        genel_toplam = g0
        recalculated = float(_to_decimal_2(net) + _to_decimal_2(kdv))
        if abs(_to_decimal_2(genel_toplam) - _to_decimal_2(recalculated)) > GOLDEN_RULE_TOL:
            genel_toplam = recalculated
            override_applied = True
        total_source = source

    if strict_trio is not None:
        _apply_trio(strict_trio, source="strict_triple_match")
    else:
        bottom_first = _lines_bottom_first(lines_y)
        bottom_nums = _bottom_number_candidates(bottom_first)
        vat_cands = _vat_candidates_from_lines(lines_y)
        combo = _solve_trio_net_vat_total(bottom_nums, vat_cands)
        if combo is not None:
            _apply_trio(combo, source="combo_golden_match")
        else:
            net = 0.0
            kdv = 0.0
            genel_toplam = None

    return {
        "firma_adi": "Tespit Edilemedi",
        "tarih": datetime.now().strftime("%Y-%m-%d"),
        "net_tutar": float(net) if net is not None else 0.0,
        "kdv_tutari": float(kdv) if kdv is not None else 0.0,
        "genel_toplam": float(genel_toplam) if genel_toplam is not None else None,
        "total_override_applied": bool(override_applied),
        "total_source": total_source,
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