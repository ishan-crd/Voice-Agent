# Start the API + dashboard (reads .env). Ctrl+C to stop.
#   .\start.ps1            foreground
#   .\start.ps1 -Tunnel    also expose it through a Cloudflare quick tunnel (needs cloudflared)
param([switch]$Tunnel)
$ErrorActionPreference = "Stop"
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

& .\.venv\Scripts\python.exe -m engine.server
