## AUM


# run all scenarios for all dir:

1. go into the dir with the file run_scenarios
2. run (in powershell)
foreach ($dir in $dirs) {
     $path = Join-Path $basePath $dir

     Write-Host "Starte: $dir" -ForegroundColor Cyan

     python .\run_scenarios.py `
         "$path" `
         --model-name "gemma4:12b" `
         --num-ctx 131072 `
         --temperature 0.2 `
         --show-thinking-in-console

     Write-Host "Fertig: $dir" -ForegroundColor Green
 }



# run all scenarios for one dir

1. go into the model dir 
2. python batch_slm_to_adl.py ./dir_for_txt_files ./dir_for_adl_files
note that dir_for_txt_files is output_slm and dir_for_adl_files is output_adl (if nothing changed)

# run visualization for every model group 
1. go into the "Modelle" dir 
2. run python time_plot.py
note that it only works if there is a valid run time report 

# run visualization for one model type
1. go into the Modelle/yourModelType dir 
2. run python visualization.py