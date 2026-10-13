# Ask Windows to stay awake (ES_SYSTEM_REQUIRED) until the final pipeline reports "all done".
# This is a per-process request, like a video player's; it changes no power settings and ends
# when this script exits. It cannot prevent sleep from closing the lid.
param([string]$DoneFile)
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Awake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f); }
"@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[void][Awake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
"keep-awake on $(Get-Date -Format T)"
while (-not ((Test-Path $DoneFile) -and (Select-String -Path $DoneFile -Pattern 'all done' -Quiet))) { Start-Sleep 60 }
[void][Awake]::SetThreadExecutionState($ES_CONTINUOUS)
"keep-awake off $(Get-Date -Format T)"
