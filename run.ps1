# Run Cancer Risk app with project venv
Set-Location $PSScriptRoot
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Host "[ERROR] .venv not found. Create it first:" -ForegroundColor Red
  Write-Host 'C:\Users\ADMIN\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv'
  Write-Host '.\.venv\Scripts\python.exe -m pip install -r requirements.txt'
  exit 1
}
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
