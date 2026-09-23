$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Установите Python 3.11 или новее и повторите запуск.' }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости.' }
}
& '.\.venv\Scripts\python.exe' -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

