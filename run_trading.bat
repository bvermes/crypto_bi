@echo off
:: Convert the notebook to a .py script
"F:\Programs\anaconda3\Scripts\jupyter-nbconvert.exe" --to script "F:\Desktop\git\crypto_bi\LiveTradingSignal.ipynb"

:: Optionally, pause or add a delay if needed
timeout /t 3

:: Now run the converted script
"F:\Programs\anaconda3\python.exe" "F:\Desktop\git\crypto_bi\LiveTradingSignal.py"
