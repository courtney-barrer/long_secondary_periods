#!/bin/bash

## scan wavelengths for image reconstruction using VLTI-Mira
# set up for gravity but could be used for other instruments

# Define range and step
start=2.000
end=2.400
step=0.002

# Calculate number of steps
total_steps=$(echo "($end - $start) / $step" | bc)
total_steps=${total_steps%.*}  # Convert to integer

# Initialize
w1=$start
i=1

export PATH="$HOME/easy-yorick/bin/:$PATH"

while (( $(echo "$w1 < $end" | bc -l) )); do
    w2=$(echo "$w1 + $step" | bc)

    # Progress message
    percent=$(echo "scale=1; 100 * $i / $total_steps" | bc)
    printf "[%3d/%3d] (%.1f%%) Running wavemin=%.3f, wavemax=%.3f\n" "$i" "$total_steps" "$percent" "$w1" "$w2"

    # Run the command
    python image_reconstruction/VLTI-Mira_image_reconstruction_pipeline.py \
        --dont_write_report \
        --ins gravity \
        --I_really_want_to_use_this_prior /home/rtc/Documents/long_secondary_periods/PMOIRED_FITS/best_models/bestparamodel_ellipse_gravity.json \
        --wavemin $w1 \
        --wavemax $w2 \
        --fov 21 \
        --mu 100 \
        --tau 1e-1 \
        --savefig /home/rtc/Documents/long_secondary_periods/image_reconstruction/image_reco/gravity_wvl_scan/v1/

    # Increment
    w1=$(echo "$w1 + $step" | bc)
    i=$((i + 1))
done
