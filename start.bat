@echo off
cd /d "%~dp0"
echo.
echo  CodeBrain - Starting server...
echo  The browser will open automatically at http://localhost:8000
echo  Keep this window open while using CodeBrain.
echo.
python src\codebrain\cli.py serve
if errorlevel 1 (
    echo.
    echo  Failed to start. Make sure Python is installed and dependencies are set up:
    echo    pip install -r requirements.txt
    echo.
    pause
)
