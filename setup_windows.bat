@echo off
REM One-time setup for Windows 10/11. Run from the project folder (double-click or in a terminal).
setlocal
where py >nul 2>nul || (echo Python launcher "py" not found. Install Python 3.11 or 3.12 from python.org and tick "Add to PATH". & exit /b 1)
if not exist venv ( py -3 -m venv venv || exit /b 1 )
call venv\Scripts\activate.bat
python -m pip install --upgrade pip || exit /b 1
pip install -r requirements.txt || exit /b 1
echo.
echo Optional, for NVIDIA GPU transcription (CUDA 12 runtime libraries from pip):
echo     pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
echo.
python scripts\check_environment.py
echo.
echo Next: install Ollama from https://ollama.com/download , then run:  python scripts\download_models.py
endlocal
