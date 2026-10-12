# Final pipeline: run after every training run has finished.
# eval (all runs) -> ensemble -> mpc -> tables -> plot data -> TikZ figures + SVG -> paper PDF
$ErrorActionPreference = "Continue"
$v2 = "D:\Research\research-workflow\examples\euler-jepa-mini\v2"
$py = "D:\Anaconda3\envs\rl-gpu\python.exe"
Set-Location $v2
function Busy { [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_v2.py (train|hybrid)' }) }
$quiet = 0
while ($quiet -lt 3) { if (Busy) { $quiet = 0 } else { $quiet++ }; Start-Sleep 60 }   # 3 consecutive idle minutes
"training+hybrid finished $(Get-Date -Format T)"
& $py -u run_v2.py eval *> logs\eval_final.out;      "eval done $(Get-Date -Format T)"
& $py -u run_v2.py ensemble *> logs\ensemble.out;    "ensemble done $(Get-Date -Format T)"
& $py -u run_v2.py mpc *> logs\mpc_final.out;        "mpc done $(Get-Date -Format T)"
$env:CUDA_VISIBLE_DEVICES = ""
& $py make_tables.py *> logs\make_tables.out
& $py make_figures.py *> logs\make_figures.out
Set-Location paper
& $py export_plot_data.py *> ..\logs\export_plot_data.out
& $py build_figures.py *> ..\logs\build_figures.out
pdflatex -interaction=nonstopmode main.tex | Out-Null; bibtex main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null; pdflatex -interaction=nonstopmode main.tex | Select-String 'Output written'
"all done $(Get-Date -Format T)"
Get-ChildItem ..\runs | Where-Object { -not (Test-Path "$($_.FullName)\config.json") } | ForEach-Object { "UNFINISHED: $($_.Name)" }
Get-Content ..\logs\eval_final.out, ..\logs\ensemble.out, ..\logs\mpc_final.out, ..\logs\build_figures.out, ..\logs\export_plot_data.out | Select-String 'Traceback|Error|FAILED|failed' -Context 0,2
