#!/usr/bin/env python3
from astropy.io import fits
import numpy as np
import argparse, os, glob


### THIS FILE IS TO ADD *BAD* FLAGS TO OIFITS FILES TO FILTER FOR SPECIFIC WAVELENGTHS! 
## WRITES NEW FITS FILES WITH THE EDITTED FLAGS IN PLACE.

ROOT = "/home/rtc/Documents/long_secondary_periods/data"

def bandmask_from_oi_wavelength(hdul, ranges_um):
    """Return {INSNAME: keep_mask} from OI_WAVELENGTH.EFF_WAVE (µm)."""
    masks = {}
    for h in hdul:
        if h.name == "OI_WAVELENGTH":
            ins = h.header.get("INSNAME")
            lam_um = np.asarray(h.data["EFF_WAVE"], float) * 1e6  # m -> µm
            keep = np.zeros_like(lam_um, dtype=bool)
            for lo, hi in ranges_um:
                keep |= (lam_um >= lo) & (lam_um <= hi)
            masks[ins] = keep
    return masks

def apply_flag_mask(hdul, masks):
    """Set FLAG[:, ~keep] = True in OI_VIS2 / OI_VIS / OI_T3 / OI_FLUX."""
    for h in hdul:
        if h.name in ("OI_VIS2", "OI_VIS", "OI_T3", "OI_FLUX"):
            ins = h.header.get("INSNAME")
            keep = masks.get(ins)
            if keep is None:
                continue
            flag = np.array(h.data["FLAG"], dtype=bool, copy=True)
            # Ensure 2D (NROW, NWAVE); OI_T3 has same shape convention
            if flag.ndim != 2 or flag.shape[1] != keep.size:
                raise ValueError(f"{h.name} FLAG shape {flag.shape} "
                                 f"doesn't match NWAVE={keep.size} for INSNAME={ins}")
            flag[:, ~keep] = True
            h.data["FLAG"] = flag

def write_band_filtered(infile, outfile, ranges_um):
    with fits.open(infile, mode="readonly") as hdul:
        hdul_out = fits.HDUList([h.copy() for h in hdul])
    masks = bandmask_from_oi_wavelength(hdul_out, ranges_um)
    if not masks:
        raise RuntimeError(f"No OI_WAVELENGTH tables found in {infile}")
    apply_flag_mask(hdul_out, masks)
    # provenance
    hdr0 = hdul_out[0].header
    for lo, hi in ranges_um:
        hdr0.add_history(f"Flagged channels outside [{lo:.2f}, {hi:.2f}] um")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    hdul_out.writeto(outfile, overwrite=True)

def main():
    ap = argparse.ArgumentParser(description="Flag OIFITS channels outside a wavelength band and save copies.")
    ap.add_argument("--ins", required=True, help="instrument key for output path (e.g. pionier, gravity, matisse_L)")
    ap.add_argument("--band", required=True, help="label for folder name (e.g. L, M)")
    ap.add_argument("--wvl-min", type=float, required=True, help="band min (µm)")
    ap.add_argument("--wvl-max", type=float, required=True, help="band max (µm)")
    ap.add_argument("--input-glob", required=True, help="glob for input FITS (e.g. '/path/*.fits')")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit("No files matched input-glob.")

    out_dir = os.path.join(ROOT,  f"{args.ins}_wvl_filtered_{args.band}")
    tag = f"wvl_filt_{args.wvl_min:.2f}-{args.wvl_max:.2f}"

    print(f"Saving to: {out_dir}")
    print(f"Band: [{args.wvl_min:.2f}, {args.wvl_max:.2f}] µm  tag: {tag}")
    rng = [(args.wvl_min, args.wvl_max)]

    for ii,f in enumerate(files):
        base = os.path.basename(f)
        if base.lower().endswith(".fits"):
            out_name = base[:-5] + f"_{tag}.fits"
        else:
            out_name = base + f"_{tag}.fits"
        out_path = os.path.join(out_dir, out_name)
        write_band_filtered(f, out_path, rng)
        print(f"  wrote {out_path}. {(100*ii/len(files)):.2f}% complete.")

if __name__ == "__main__":
    main()


"""
Matisse files combine L and M band
#e.g. to filter L band 
############################
python wvl_filter_oifits.py \
  --ins matisse --band L \
  --wvl-min 3.30 --wvl-max 3.60 \
  --input-glob "/home/rtc/Documents/long_secondary_periods/data/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits"

#e.g. to filter M band 
############################
python wvl_filter_oifits.py \
  --ins matisse --band M \
  --wvl-min 4.60 --wvl-max 4.90 \
  --input-glob "/home/rtc/Documents/long_secondary_periods/data/matisse/reduced_calibrated_data_1/all_chopped_L/*.fits"

"""