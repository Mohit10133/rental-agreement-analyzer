@echo off
echo Starting SafeLeaseX Application...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo Error: Could not activate virtual environment
    echo Please make sure the virtual environment is properly set up
    pause
    exit /b 1
)

echo Virtual environment activated successfully!
echo.

REM Install/update dependencies if needed
echo Checking dependencies...
pip install -r requirements.txt --quiet

REM Start the Flask application
echo Starting Flask server...
echo.
echo SafeLeaseX will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
