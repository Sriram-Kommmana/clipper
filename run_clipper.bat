@echo off
REM Starts the review UI at http://127.0.0.1:7860 (local only).
call venv\Scripts\activate.bat || (echo Run setup_windows.bat first. & exit /b 1)
python app.py %*
