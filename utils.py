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

_NET_PATTERNS = [
    # KDV Matrahı (en güvenilir net kaynağı)
    re.compile(r"(?:kdv\s*matrahı?|vergi\s*matrahı?|matrah)[^:]{0,45}:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
    re.compile(r"(?:net\s*tutar)[^:]{0,45}:\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
    re.compile(r"mal\s*hizmet\s*toplam\s*tutar[ıi]?\s*[:\s]\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", re.I),
]

_KDV_PATTERNS = [
    # Hesaplanan KDV (en güvenilir KDV kaynağı)
    re.compile(r"hesaplanan\s*kdv\s*\([^)]*\)\s*:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv\s*\([^)]*\)\s+(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv[^:]{0,40}:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"kdv\s*tutar[ıi][^:]{0,30}:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"(?:katma\s*değer|vergi\s*tutar(?:ı|i))[^:]{0,30}:\s*(\d+(?:[.,]\d{2})?)", re.I),
]

# Sadece %20 KDV için özel desenler (öncelikli)
_KDV_20_PATTERNS = [
    re.compile(r"hesaplanan\s*kdv\s*\(%20[.,]?\d*\)\s*:\s*(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"hesaplanan\s*kdv\s*\(%20[.,]?\d*\)\s+(\d+(?:[.,]\d{2})?)", re.I),
    re.compile(r"(?:%|oran[:\s]*)\s*20[^:]{0,30}:\s*(\d+(?:[.,]\d{2})?)", re.I),
]

_MONEY_IN_LINE = re.compile(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2}")

# Ürün satırı işaretçisi — bu satırlardaki tutarlar toplam adayı sayılmaz
_PRODUCT_ROW_HINT = re.compile(r"\badet\b|\bpcs\b|\bbirim\s*fiyat\b", re.I)
# İskonto sonrası net matrahı kesin işaret eden satırlar
_NET_MATRAH_HINT = re.compile(
    r"kdv\s*matrah|vergi\s*matrah|vergi\s*hari[cç]|matrah\s*\(", re.I
)
# Kesin toplam satırları
_HARD_TOTAL_HINT = re.compile(r"ödenecek\s*tutar|vergiler\s*dahil", re.I)


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
    # OCR harf→rakam düzeltmesi
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


def _is_ara_toplam_line(text: str) -> bool:
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
    n_lines = len(lines_bottom_first)

    for ln in lines_bottom_first:
        tx = ln["text"]
        if _is_discount_line(tx):
            continue
        if bool(_PRODUCT_ROW_HINT.search(tx)):
            continue
        y = float(ln["y"])
        skip_subtotal_row = _is_ara_toplam_line(tx)
        for pat, pri in _TOTAL_KEYWORD_PATTERNS:
            if skip_subtotal_row:
                continue
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

    if n_lines:
        cutoff = max(0, n_lines * 2 // 3)
        for ln in lines_bottom_first[: max(15, n_lines - cutoff)]:
            tx = ln["text"]
            if _is_discount_line(tx) or _is_ara_toplam_line(tx):
                continue
            if bool(_PRODUCT_ROW_HINT.search(tx)):
                continue
            for m in _MONEY_IN_LINE.finditer(tx):
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
    Iskontolu faturalar dahil doğru Net/KDV/Toplam tespiti.
    Öncelik sırası:
    1. Context-based extraction: KDV Matrahı (net) + Hesaplanan KDV (kdv) + Ödenecek Tutar (total)
    2. Keyword-anchored extraction
    3. Net+KDV=Toplam golden rule eşleşmesi
    4. Fallback
    """
    lines_y = _parse_ocr_lines(ocr_lines)
    if not lines_y:
        lines_y = _lines_from_text(text)

    bottom_first = _lines_bottom_first(lines_y)
    
    # ============ 1. CONTEXT-BASED EXTRACTION (EN GÜVENİLİR) ============
    net_from_context = None
    kdv_from_context = None
    total_from_context = None
    
    for ln in lines_y:
        tx = _normalize_whitespace(ln["text"])
        tx_lower = tx.lower()
        
        # KDV Matrahı (NET TUTAR) - en güvenilir net kaynağı
        if re.search(r"kdv\s*matrahı?|vergi\s*matrahı?|matrah", tx_lower):
            # Sadece sayıyı bul
            match = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx)
            if match:
                val = parse_tr_number(match.group(1))
                if val and val > 0:
                    net_from_context = val
                    logger.debug(f"Net from KDV matrah: {val}")
        
        # Hesaplanan KDV (KDV TUTARI) - en güvenilir KDV kaynağı
        if re.search(r"hesaplanan\s*kdv", tx_lower):
            # Satır içinde sayı var mı?
            match = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx)
            if match:
                val = parse_tr_number(match.group(1))
                if val and val > 0:
                    kdv_from_context = val
                    logger.debug(f"KDV from Hesaplanan KDV (same line): {val}")
            else:
                # Sonraki satırda sayı ara (bazen KDV değeri alt satırda olabilir)
                idx = lines_y.index(ln)
                for j in range(idx + 1, min(idx + 3, len(lines_y))):
                    next_tx = lines_y[j]["text"]
                    match_next = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", next_tx)
                    if match_next:
                        val = parse_tr_number(match_next.group(1))
                        if val and val > 0 and len(next_tx.strip()) < 30:  # KDV satırı genelde kısa olur
                            kdv_from_context = val
                            logger.debug(f"KDV from next line: {val}")
                            break
        
        # Ödenecek Tutar (GENEL TOPLAM) - en güvenilir toplam kaynağı
        if re.search(r"ödenecek\s*tutar|vergiler\s*dahil", tx_lower):
            match = re.search(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})", tx)
            if match:
                val = parse_tr_number(match.group(1))
                if val and val > 0:
                    total_from_context = val
                    logger.debug(f"Total from Odenecek Tutar: {val}")
    
    # Context-based extraction başarılıysa doğrudan kullan
    if net_from_context and kdv_from_context and total_from_context:
        # Doğrulama: Net + KDV = Total?
        net_dec = _to_decimal_2(net_from_context)
        kdv_dec = _to_decimal_2(kdv_from_context)
        total_dec = _to_decimal_2(total_from_context)
        
        if abs(net_dec + kdv_dec - total_dec) <= GOLDEN_RULE_TOL:
            logger.info(f"Context extraction successful: Net={net_from_context}, KDV={kdv_from_context}, Total={total_from_context}")
            return {
                "firma_adi": "Tespit Edilemedi",
                "tarih": datetime.now().strftime("%Y-%m-%d"),
                "net_tutar": float(net_from_context),
                "kdv_tutari": float(kdv_from_context),
                "genel_toplam": float(total_from_context),
                "total_override_applied": False,
                "total_source": "context_based",
            }
    
    # ============ 2. KEYWORD-ANCHORED EXTRACTION ============
    # Net ve KDV adaylarını topla (ürün satırlarını atla)
    net_hits = _collect_pattern_amounts(lines_y, _NET_PATTERNS, skip_product_rows=True)
    
    # Önce %20 KDV desenlerini dene (daha spesifik)
    kdv_20_hits = _collect_pattern_amounts(lines_y, _KDV_20_PATTERNS, skip_product_rows=True)
    # Genel KDV desenleri
    kdv_general_hits = _collect_pattern_amounts(lines_y, _KDV_PATTERNS, skip_product_rows=True)
    
    # KDV adaylarını birleştir, %20 KDV olanlara öncelik ver
    kdv_hits = list(kdv_20_hits)
    seen_kdv = {round(v, 2) for v, _ in kdv_hits}
    for kv, ky in kdv_general_hits:
        if round(kv, 2) not in seen_kdv:
            seen_kdv.add(round(kv, 2))
            kdv_hits.append((kv, ky))
    
    # Net değerleri KDV adaylarından filtrele (net değer KDV olamaz)
    net_vals = {round(v, 2) for v, _ in net_hits}
    kdv_hits_filtered = [(v, y) for v, y in kdv_hits if round(v, 2) not in net_vals]
    kdv_hits_for_pairs = kdv_hits_filtered if kdv_hits_filtered else kdv_hits
    
    # Toplam adaylarını bul
    candidates = _grand_total_candidates(bottom_first)
    
    # Kesin toplam anchor'larını önce dene (pri>=98: Ödenecek Tutar, Vergiler Dahil)
    hard_total_lines = [(v, pri, y) for v, pri, y in candidates if pri >= 98]
    search_order = hard_total_lines + [c for c in candidates if c[1] < 98]
    
    net = 0.0
    kdv = 0.0
    genel_toplam = None
    chosen = False
    
    # Net ve KDV eşleştirmesi yap
    pairs = _pick_net_kdv_for_math(net_hits, kdv_hits_for_pairs)
    
    for total_guess, _pri, _y in search_order:
        td = _to_decimal_2(total_guess)
        for nv, kv in pairs:
            if abs(_to_decimal_2(nv) + _to_decimal_2(kv) - td) <= GOLDEN_RULE_TOL:
                genel_toplam = float(total_guess)
                net = float(nv)
                kdv = float(kv)
                chosen = True
                break
        if chosen:
            break
    
    # Fallback: pairs toplamını toplam olarak kullan
    if not chosen and pairs:
        ref_n, ref_k = pairs[0]
        ref_sum = float(_to_decimal_2(ref_n) + _to_decimal_2(ref_k))
        for total_guess, _pri, _y in search_order:
            if abs(_to_decimal_2(total_guess) - _to_decimal_2(ref_sum)) <= GOLDEN_RULE_TOL:
                genel_toplam = float(total_guess)
                net = float(ref_n)
                kdv = float(ref_k)
                chosen = True
                break
    
    if not chosen and pairs:
        ref_n, ref_k = pairs[0]
        genel_toplam = float(_to_decimal_2(ref_n) + _to_decimal_2(ref_k))
        net = float(ref_n)
        kdv = float(ref_k)
        chosen = True
    
    # KDV bulunamadı ama Net bulundu: Toplam'dan KDV'yi hesapla
    if not chosen and net_hits:
        best_net = net_hits[0][0]
        for total_guess, pri, _y in search_order:
            if pri >= 70 and total_guess > best_net:
                derived_kdv = float(_to_decimal_2(total_guess) - _to_decimal_2(best_net))
                if derived_kdv > 0:
                    net = best_net
                    kdv = derived_kdv
                    genel_toplam = float(total_guess)
                    chosen = True
                    break
    
    # Son çare: en alttaki sayıyı toplam olarak al
    if not chosen:
        bottom_last = _extract_last_currency_value(bottom_first)
        if bottom_last is not None:
            genel_toplam = float(bottom_last)
            chosen = True
    
    # Net+KDV her zaman kazanır - toplamı hesapla ve doğrula
    if net > 0 and kdv > 0:
        calculated = float(_to_decimal_2(net) + _to_decimal_2(kdv))
        if genel_toplam is None or abs(_to_decimal_2(genel_toplam) - _to_decimal_2(calculated)) > GOLDEN_RULE_TOL:
            genel_toplam = calculated
    
    if genel_toplam is None:
        net = 0.0
        kdv = 0.0
    
    logger.info(f"Extraction result: Net={net}, KDV={kdv}, Total={genel_toplam}")
    
    return {
        "firma_adi": "Tespit Edilemedi",
        "tarih": datetime.now().strftime("%Y-%m-%d"),
        "net_tutar": float(net) if net is not None else 0.0,
        "kdv_tutari": float(kdv) if kdv is not None else 0.0,
        "genel_toplam": float(genel_toplam) if genel_toplam is not None else None,
        "total_override_applied": False,
        "total_source": "keyword_anchor",
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
