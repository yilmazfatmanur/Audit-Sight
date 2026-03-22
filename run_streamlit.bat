@echo off
cd /d "%~dp0"
echo AuditSight Pro baslatiliyor...
echo Klasor: %CD%
python -m pip install -q -r requirements.txt 2>nul
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo HATA: Yukaridaki mesaji okuyun. Sik gorulen cozumler:
    echo   pip install streamlit
    echo   Bu dosyayi Audit-Sight klasorunden calistirdiginizdan emin olun.
    pause
)
