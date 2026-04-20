@echo off
setlocal

cd /d "%~dp0"

echo Starting S&P 500 Behavioral Finance Stock Screener...
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python was not found on PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

python -m pip show streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing required packages. This may take a minute...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo Package installation failed. Please check your internet connection.
        pause
        exit /b 1
    )
)

echo.
echo Opening the site at http://localhost:8501
echo Keep this window open while using the app.
echo.

start http://localhost:8501
python -m streamlit run app.py

pause
