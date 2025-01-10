#!/bin/bash
#export PATH="$HOME/easy-yorick/bin/:$PATH"
export PATH=$HOME/easy-yorick/bin:$PATH

###### before running activate environment : source .venv/bin/activate

# Define the options for --ins
#ins_options=("pionier" "gravity" "matisse_L" "matisse_M" "matisse_N_8um" "matisse_N_9um" "matisse_N_10um" "matisse_N_11um" "matisse_N_12um")

# Define the options for --ins .. 
ins_options=("pionier" "gravity" "matisse_L" "matisse_M" "matisse_N_8.0um" "matisse_N_8.5um" "matisse_N_9.0um" "matisse_N_9.5um" "matisse_N_10.0um" "matisse_N_10.5um" "matisse_N_11.0um" "matisse_N_11.5um" "matisse_N_12.0um" "matisse_N_12.5um")
#("gravity_line_CO2-0" "gravity_line_CO3-1" "gravity_line_CO4-2")
# Base save directory
base_save_dir="/home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco"

# also can use Dirac
prior="UD"
mu=500
tau=100
# Loop through each option and run the Python script with appropriate parameters
for ins in "${ins_options[@]}"; do
    save_dir="${base_save_dir}/${ins}/"

    if [ "$ins" == "pionier" ]; then
        # Special options for pionier
        python image_reconstruction/VLTI-Mira_image_reconstruction_pipeline.py --ins "$ins" --prior "$prior" --mu "$mu" --tau "$tau" --plot_image_logscale --plot_logV2 --savefig "$save_dir" 
    else
        # Default options for other values of --ins
        python image_reconstruction/VLTI-Mira_image_reconstruction_pipeline.py --ins "$ins" --prior "$prior" --mu "$mu" --tau "$tau" --plot_image_logscale --savefig "$save_dir" 
    fi
done


#python image_reconstruction/VLTI-Mira_image_reconstruction_pipeline.py --ins matisse_N_9.5um --I_really_want_to_use_this_prior /home/rtc/Documents/long_secondary_periods/PMOIRED_FITS/best_models/bestparamodel_disk_N_mid.json --fov 400 --pixelsize 20 --mu 10000 --tau 1e-5 --plot_image_logscale --savefig /home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/matisse_N_9.5um/