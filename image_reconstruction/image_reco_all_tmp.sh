#!/bin/bash

# Define variables
fov=90

# Define instruments and prior lists
IR_ins_list=("pionier" "gravity" "matisse_L" "matisse_M")
N_ins_list=("matisse_N_8.0um" "matisse_N_8.5um" "matisse_N_9.0um" "matisse_N_9.5um" \
           "matisse_N_10.0um" "matisse_N_10.5um" "matisse_N_11.0um" "matisse_N_11.5um" \
           "matisse_N_12.0um" "matisse_N_12.5um")

instruments=("${IR_ins_list[@]}" "${N_ins_list[@]}")
IR_prior_list=("bestparamodel_ellipse_gravity.json" "bestparamodel_ellipse_gravity.json" \
               "bestparamodel_binary_matisse_L.json" "bestparamodel_binary_matisse_M.json")
N_prior_list=("bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json" \
             "bestparamodel_disk_matisse_N_short_8.5um.json")
prior_list=("${IR_prior_list[@]}" "${N_prior_list[@]}")

wvls=("1.6" "2.2" "3.5" "4.7" "8.0" "8.5" "9.0" "9.5" "10.0" "10.5" "11.0" "11.5" "12.0" "12.5")

# Create instrument dictionary
mu=(1000.0 1000.0 10 10 10 10 10 10 10 10 10 10 10 10)
tau=(1e-5 1e-1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)

# Iterate over instruments
for i in "${!wvls[@]}"; do
    wvl="${wvls[i]}"
    instrument="${instruments[i]}"
    prior="${prior_list[i]}"
    mu_val="${mu[i]}"
    tau_val="${tau[i]}"


    if [[ "${instrument}" == *"matisse_N"* ]]; then
        cmd="python image_reconstruction/VLTI-Mira_image_reconstruction_pipeline.py \
            --ins ${instrument} \
            --I_really_want_to_use_this_prior /home/rtc/Documents/long_secondary_periods/PMOIRED_FITS/best_models/${prior} \
            --mu ${mu_val} \
            --tau ${tau_val} \
            --plot_image_logscale \
            --savefig /home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/FINAL/${instrument}/"
            
    else
        cmd="python image_reconstruction/VLTI-Mira_image_reconstruction_pipeline.py \
            --ins ${instrument} \
            --prior UD \
            --mu ${mu_val} \
            --tau ${tau_val} \
            --savefig /home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/FINAL/${instrument}/"
            
    fi

    echo "Running: ${cmd}"
    eval ${cmd}
done
