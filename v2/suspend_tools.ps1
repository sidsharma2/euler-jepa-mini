# Suspend / resume whole processes (Windows NtSuspendProcess / NtResumeProcess).
# Usage: suspend_tools.ps1 -Action suspend|resume -Pids 123,456
param([ValidateSet("suspend", "resume")] [string]$Action, [int[]]$Pids)
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ProcCtl {
    [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr h);
    [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);
    [DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(int access, bool inherit, int pid);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
"@
foreach ($p in $Pids) {
    $h = [ProcCtl]::OpenProcess(0x0800, $false, $p)   # PROCESS_SUSPEND_RESUME
    if ($h -eq [IntPtr]::Zero) { "could not open $p"; continue }
    $rc = if ($Action -eq "suspend") { [ProcCtl]::NtSuspendProcess($h) } else { [ProcCtl]::NtResumeProcess($h) }
    [void][ProcCtl]::CloseHandle($h)
    "$Action $p -> status $rc"
}
