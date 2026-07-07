# AUM

1. go into the dir with the file run_scenarios
2. run (in powershell)
foreach ($folder in $folders) {
     $path = Join-Path $basePath $folder

     Write-Host "Starte: $folder" -ForegroundColor Cyan

     python .\run_scenarios.py `
         "$path" `
         --model-name "gemma4:12b" `
         --num-ctx 131072 `
         --temperature 0.2 `
         --show-thinking-in-console

     Write-Host "Fertig: $folder" -ForegroundColor Green
 }
