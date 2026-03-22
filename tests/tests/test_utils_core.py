"""Çekirdek mantık testleri (OCR/Gemini ağı yok)."""
from __future__ import annotations

import pytest

from utils import (
    extract_invoice_fields,
    infer_invoice_category_heuristic,
    parse_tr_number,
    validate_golden_rule,
)


class TestParseTrNumber:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.234,56", 1234.56),
            ("763,00", 763.0),
            ("80,97 TL", 80.97),
            ("100", 100.0),
            ("", None),
        ],
    )
    def test_parse(self, raw, expected):
        got = parse_tr_number(raw)
        if expected is None:
            assert got is None
        else:
            assert got is not None and abs(float(got) - expected) < 1e-6


class TestValidateGoldenRule:
    def test_ok_exact(self):
        r = validate_golden_rule(100.0, 18.0, 118.0)
        assert r["ok"] is True
        assert r["box"] == "green"

    def test_ok_within_line_tolerance(self):
        # Çıkarımdan gelen ~1 TL sapma
        r = validate_golden_rule(100.0, 18.0, 118.9)
        assert r["ok"] is True

    def test_missing_field(self):
        r = validate_golden_rule(None, 10.0, 110.0)
        assert r["ok"] is None
        assert r["box"] == "gray"

    def test_fail_large_gap(self):
        r = validate_golden_rule(100.0, 18.0, 200.0)
        assert r["ok"] is False
        assert r["box"] == "red"
        assert "Fark" in (r.get("message") or "")


class TestExtractInvoiceFields:
    def test_sample_invoice_text(self):
        text = """
        VITISFERA
        KDV Matrahı ( %1.00): 755,45
        Hesaplanan KDV (%1.00) 7,55
        Vergiler Dahil Toplam Tutar: 763,00
        Odenecek Tutar: 763,00
        """
        fields = extract_invoice_fields(text, None)
        assert fields.get("genel_toplam") is not None
        assert abs(float(fields["genel_toplam"]) - 763.0) < 0.01
        # net + kdv ≈ toplam
        n = fields.get("net_tutar")
        k = fields.get("kdv_tutari")
        g = fields.get("genel_toplam")
        vr = validate_golden_rule(n, k, g)
        assert vr["ok"] is True, f"Expected pass, got {vr}"


class TestHeuristicCategory:
    def test_fuel(self):
        h = infer_invoice_category_heuristic("shell akaryakıt istasyon")
        assert "Akaryakıt" in (h.get("category_tr") or "")


class TestGeminiModelName:
    """Gemini modeli kodda tutarlı kullanılıyor mu (derleme seviyesi)."""

    def test_utils_imports(self):
        import utils as u

        assert hasattr(u, "gemini_invoice_enrichment")
