# Start the API + dashboard (reads .env) and open the console in your browser.
#   .\start.ps1            foreground; Ctrl+C stops it
#   .\start.ps1 -Tunnel    also expose it through a Cloudflare quick tunnel (needs cloudflared)
#   .\start.ps1 -NoBrowser don't open the browser
param([switch]$Tunnel, [switch]$NoBrowser)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")

if ($Tunnel) {
    if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
        winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements --silent
        $env:PATH = [Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [Environment]::GetEnvironmentVariable("PATH", "User")
    }
    Start-Process cloudflared -ArgumentList "tunnel", "--url", "http://localhost:8000" -NoNewWindow
    Write-Host "cloudflared prints a https://*.trycloudflare.com URL above - that is your public base URL" -ForegroundColor Yellow
}

# already running? just open the console
try {
    $null = Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 2
    Write-Host "server already running on :8000" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process "http://localhost:8000/app/" }
    exit 0
} catch {}

if (-not $NoBrowser) {
    # open the console once the models are warm (first start downloads weights and can take a few minutes)
    Start-Job -ScriptBlock {
        for ($i = 0; $i -lt 600; $i++) {
            Start-Sleep 2
            try { $null = Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 2; Start-Process "http://localhost:8000/app/"; break } catch {}
        }
    } | Out-Null
}

Write-Host "starting Voice-Agent on http://localhost:8000 (console at /app) - loading models..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m engine.server
