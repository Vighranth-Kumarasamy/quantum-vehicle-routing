@echo off
title Quantum Vehicle Routing
echo ==============================================
echo   Quantum-Optimized Delivery Routes
echo ==============================================
echo.
echo Installing/updating required packages...
python -m pip install -r requirements.txt
echo.
echo Starting the application...
python -m streamlit run app.py
pause
