# One-shot setup on Windows: Python 3.11 venv, CUDA torch, chatterbox, ffmpeg, node, dashboard build.
# Run from the repo root in PowerShell:  .\setup.ps1
$ErrorActionPreference = "Stop"

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Write-Host "== Python 3.11" -ForegroundColor Cyan
if (-not (py -3.11 --version 2>$null)) {
    if (Have py) { py install 3.11 } else { winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements }
}
if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet

Write-Host "== CUDA torch (must come before chatterbox so its pin resolves to the CUDA wheel)" -ForegroundColor Cyan
& $py -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

Write-Host "== Engine dependencies" -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

Write-Host "== ffmpeg / node (for mp3+opus output and the dashboard)" -ForegroundColor Cyan
if (-not (Have ffmpeg)) { winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements --silent }
if (-not (Have node))   { winget install --id OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements --silent }
$env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")

Write-Host "== Dashboard" -ForegroundColor Cyan
Push-Location web
npm install --no-audit --no-fund
npm run build
Pop-Location

Write-Host "== GPU clocks" -ForegroundColor Cyan
# A voice server is a bursty workload: the driver drops the clocks between requests and
# ramps them back per request, which costs 200-400 ms of first-audio latency and makes
# Whisper 2-3x slower. A logon task locks the clocks (needs one UAC prompt; ~40 W extra idle).
$nv = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if ($nv) {
    $lock = "schtasks /Create /F /RU SYSTEM /SC ONLOGON /RL HIGHEST /TN VoiceAgentGpuClocks /TR `"\`"$nv\`" -lgc 1695,1830`" & schtasks /Run /TN VoiceAgentGpuClocks"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $lock -Verb RunAs -Wait
    Write-Host "clock lock task created (undo: schtasks /Delete /TN VoiceAgentGpuClocks /F ; nvidia-smi -rgc)"
}

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host "created .env from .env.example - edit it before going public" -ForegroundColor Yellow }

Write-Host ""
Write-Host "Verify:  $py -c `"import torch;print(torch.cuda.is_available())`"" -ForegroundColor Green
Write-Host "Smoke:   $py scripts\smoke_test.py --models turbo" -ForegroundColor Green
Write-Host "Run:     .\start.ps1" -ForegroundColor Green
