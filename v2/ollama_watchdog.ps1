# Safe idle-unload for Ollama while the Euler-JEPA runs need the GPU (approved by Sid).
# Every minute, only when NO client is connected to Ollama (no request in flight):
#   * a loaded model is asked to unload with a plain `ollama stop`;
#   * a runner stuck in "Stopping..." on two consecutive idle checks (Ollama bug) is ended.
# A request that is running is never interrupted. Exits when the final pipeline reports "all done".
param([string]$DoneFile)
function ClientConnected {
    [bool](Get-NetTCPConnection -RemotePort 11434 -State Established -ErrorAction SilentlyContinue)
}
$stuckIdle = 0
while (-not ((Test-Path $DoneFile) -and (Select-String -Path $DoneFile -Pattern 'all done' -Quiet))) {
    if (ClientConnected) { $stuckIdle = 0 }
    else {
        $ps = ollama ps 2>$null | Select-Object -Skip 1 | Where-Object { $_ }
        if (-not $ps) { $stuckIdle = 0 }
        foreach ($line in $ps) {
            $name = ($line -split '\s+')[0]
            if ($line -match 'Stopping') {
                $stuckIdle++
                if ($stuckIdle -ge 2 -and -not (ClientConnected)) {
                    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
                    "$(Get-Date -Format T) ended runner stuck in Stopping (idle, no client)"; $stuckIdle = 0
                }
            } elseif ($name -and -not (ClientConnected)) {
                ollama stop $name 2>$null | Out-Null; "$(Get-Date -Format T) unloaded idle $name"
            }
        }
    }
    Start-Sleep 60
}
"watchdog exit $(Get-Date -Format T)"
