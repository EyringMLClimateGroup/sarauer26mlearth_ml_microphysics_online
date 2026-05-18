######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: vertical_coarse.py                                                                       #
######################################################################################################
# In this script we perform vertical coarse-graining of our data                                     #
# We need to make sure that the vertical structure is properly represented                           #
######################################################################################################

import xarray as xr
import numpy as np
import glob
import os

def dbg_stats(tag, a, *, name="array"):
    a = np.asarray(a)
    finite = np.isfinite(a)
    nan = np.isnan(a).sum()
    fin = finite.sum()
    zeros = (finite & (a == 0)).sum()
    neg = (finite & (a < 0)).sum()
    pos = (finite & (a > 0)).sum()
    mn = float(np.nanmin(a)) if fin else np.nan
    mx = float(np.nanmax(a)) if fin else np.nan
    print(f"[{tag}] {name}: shape={a.shape} nan={int(nan)} finite={int(fin)} "
          f"zeros={int(zeros)} neg={int(neg)} pos={int(pos)} min={mn} max={mx}")
def dbg_new_zeros(tag, before, after, *, name="array"):
    before = np.asarray(before)
    after = np.asarray(after)
    bfin = np.isfinite(before)
    afin = np.isfinite(after)
    new_fin = (~bfin) & afin
    new_zeros = new_fin & (after == 0)
    print(f"[{tag}] {name}: newly_finite={int(new_fin.sum())} newly_zero={int(new_zeros.sum())}")


# Conservative vertical remapping
def coarse_grain_vectorized(
    dataset_input,
    z_top_input,
    z_bot_input,
    z_top_output,
    z_bot_output,
    mask_ref=None,
    fill_value=0.0,
    debug_prefix="CG",
    debug_levels=5,
):
    n_in, n_cells = dataset_input.shape
    n_out = z_top_output.shape[0]

    out = np.full((n_out, n_cells), np.nan)

    dbg_stats(f"{debug_prefix}-IN", dataset_input, name="dataset_input")
    if mask_ref is not None:
        dbg_stats(f"{debug_prefix}-MASK", mask_ref, name="mask_ref (NaN=invalid)")

    data = dataset_input.copy()

    # Interpolation diagnostics (gap filling, but ONLY on valid mask_ref points) ----
    before_interp = data.copy()

    # Fill vertical gaps column-wise, but do NOT interpolate across invalid (mask_ref) regions.
    for c in range(n_cells):
        col = data[:, c]

        if mask_ref is not None:
            valid = np.isfinite(mask_ref[:, c]) & np.isfinite(col)
        else:
            valid = np.isfinite(col)

        if valid.any():
            x = np.arange(n_in)
            # IMPORTANT: left/right NaN so we do not extrapolate beyond valid segment
            col[:] = np.interp(x, x[valid], col[valid], left=np.nan, right=np.nan)

    dbg_stats(f"{debug_prefix}-POSTINT", data, name="data after interp")
    dbg_new_zeros(f"{debug_prefix}-POSTINT", before_interp, data, name="data after interp")

    # Conservative overlap remap
    total_weight_zeros = 0
    for i in range(n_out):
        zt_o = z_top_output[i]
        zb_o = z_bot_output[i]

        weight = np.zeros(n_cells)
        accum = np.zeros(n_cells)

        for j in range(n_in):
            zt_i = z_top_input[j]
            zb_i = z_bot_input[j]

            overlap = np.maximum(np.minimum(zt_i, zt_o) - np.maximum(zb_i, zb_o), 0.0)

            if mask_ref is not None:
                valid = np.isfinite(mask_ref[j]) & np.isfinite(data[j])
            else:
                valid = np.isfinite(data[j])

            w = overlap * valid.astype(float)
            accum += w * data[j]
            weight += w

        ok = weight > 0
        out[i, ok] = accum[ok] / weight[ok]

        wz = int((weight == 0).sum())
        total_weight_zeros += wz

        if i < debug_levels:
            print(f"[{debug_prefix}-W] level={i} weight: min={float(weight.min())} "
                  f"mean={float(weight.mean())} max={float(weight.max())} zeros={wz} ok={int(ok.sum())}")

    print(f"[{debug_prefix}-WZ] total weight==0 occurrences across all target levels: {total_weight_zeros}")

    # Final fill
    if fill_value is not None:
        before_fill = out.copy()
        n_to_fill = int(np.isnan(out).sum())
        out[np.isnan(out)] = fill_value
        print(f"[{debug_prefix}-FILL] filled NaNs count={n_to_fill} with fill_value={fill_value}")
        dbg_new_zeros(f"{debug_prefix}-FILL", before_fill, out, name="out after fill")

    dbg_stats(f"{debug_prefix}-OUT", out, name="final out")
    return out


# Ensure TOP → BOTTOM ordering using GEOMETRY ONLY
def ensure_top_to_bottom(z):
    if np.nanmean(z[0]) < np.nanmean(z[-1]):
        return z[::-1]
    return z
 
# Main processing
def process_file(input_file, output_dir, grid_file, target_grid_file):
    print(f"\n Processing {input_file} ")

    ds = xr.open_dataset(input_file)
    grid_ds = xr.open_dataset(grid_file)
    tgt_ds = xr.open_dataset(target_grid_file)

    # DIMS debug
    print("[D1] ds.sizes:", dict(ds.sizes))
    print("[D2] grid_ds.sizes:", dict(grid_ds.sizes))
    print("[D3] tgt_ds.sizes:", dict(tgt_ds.sizes))

    n_cells = ds.sizes["cell"]
    n_file_lev = ds.sizes["height"]

    # Load grids
    zghalf = ensure_top_to_bottom(grid_ds["zghalf"].values)
    zhalf_tgt = ensure_top_to_bottom(tgt_ds["zhalf"].values)

    print("[Z1] zghalf shape:", zghalf.shape, "nan:", int(np.isnan(zghalf).sum()))
    print("[Z2] zhalf_tgt shape:", zhalf_tgt.shape, "nan:", int(np.isnan(zhalf_tgt).sum()))

    # Full-level heights (for debug only)
    zfull_src = 0.5 * (zghalf[:-1] + zghalf[1:])
    zfull_tgt = 0.5 * (zhalf_tgt[:-1] + zhalf_tgt[1:])

    print(f"[S1] source z range: {np.nanmax(zfull_src):.1f} .. {np.nanmin(zfull_src):.1f}")
    print(f"[S2] target z range: {np.nanmax(zfull_tgt):.1f} .. {np.nanmin(zfull_tgt):.1f}")

    n_src = zfull_src.shape[0]
    n_tgt = zfull_tgt.shape[0]

    print(f"[S3] cells={n_cells}, file levels={n_file_lev}, source levels={n_src}, target levels={n_tgt}")

    # The file contains the LOWEST levels
    zfile = zfull_src[-n_file_lev:]

    # Determine orientation of FILE using geometry
    if np.nanmean(zfile[0]) < np.nanmean(zfile[-1]):
        file_is_bottom_to_top = True
        print("[S4] file vertical order = bottom → top (will flip)")
    else:
        file_is_bottom_to_top = False
        print("[S4] file vertical order = top → bottom")

    source_top = zghalf[:-1]
    source_bot = zghalf[1:]
    target_top = zhalf_tgt[:-1]
    target_bot = zhalf_tgt[1:]

    # Geometry checks
    src_mean = np.nanmean(zghalf, axis=1) if zghalf.ndim == 2 else zghalf
    tgt_mean = np.nanmean(zhalf_tgt, axis=1) if zhalf_tgt.ndim == 2 else zhalf_tgt
    print("[Z3] src half-level monotonic (decreasing):", bool(np.all(np.diff(src_mean) <= 0)))
    print("[Z4] tgt half-level monotonic (decreasing):", bool(np.all(np.diff(tgt_mean) <= 0)))
    print("[Z5] src_mean first/last:", float(src_mean[0]), float(src_mean[-1]))
    print("[Z6] tgt_mean first/last:", float(tgt_mean[0]), float(tgt_mean[-1]))

    print("[OV1] source top max:", float(np.nanmax(source_top)), "target top max:", float(np.nanmax(target_top)))
    print("[OV2] target levels above source-top:",
          int(np.sum(np.nanmean(target_top, axis=1) > np.nanmax(source_top))))

    src_thk = source_top - source_bot
    tgt_thk = target_top - target_bot
    print("[G1] src thickness stats (min/mean/max):",
          float(np.nanmin(src_thk)), float(np.nanmean(src_thk)), float(np.nanmax(src_thk)))
    print("[G2] tgt thickness stats (min/mean/max):",
          float(np.nanmin(tgt_thk)), float(np.nanmean(tgt_thk)), float(np.nanmax(tgt_thk)))
    print("[G3] src thickness <=0 count:", int(np.sum(src_thk <= 0)))
    print("[G4] tgt thickness <=0 count:", int(np.sum(tgt_thk <= 0)))

    print("[F1] n_src:", n_src, "n_file_lev:", n_file_lev, "n_tgt:", n_tgt)
    print("[F2] zfile shape:", zfile.shape,
          "zfile mean first/last:", float(np.nanmean(zfile[0])), float(np.nanmean(zfile[-1])))

    variables = (
        [
            "tend_ta_mig", "tend_qhus_mig", "tend_qclw_mig",
            "tend_qcli_mig", "tend_qr_mig", "tend_qs_mig",
            "tend_qg_mig",
        ]
        if "tendencies" in input_file
        else
        [
            "dz_mig", "rho_mig", "pf_mig", "cpair_mig",
            "ta_mig", "qv_mig", "qc_mig", "qi_mig",
            "qr_mig", "qs_mig", "qg_mig",
        ]
    )

    ds_out = xr.Dataset(
        coords=dict(
            time=ds.time,
            cell=ds.cell,
            height=np.arange(n_tgt),  # dimension only
        )
    )

    # Loop over variables
    for v in variables:
        if v not in ds:
            continue

        print(f"\n[V] {v}")
        arr = ds[v].values  # (time, height, cell)

        print("[V0] arr shape:", arr.shape,
              "nan:", int(np.isnan(arr).sum()),
              "min:", float(np.nanmin(arr)), "max:", float(np.nanmax(arr)))
        print("[V0b] arr==0 count:", int(np.sum(arr == 0)), "finite count:", int(np.isfinite(arr).sum()))

        if v == "ta_mig":
            bad = arr == 0.0
            print(f"[V1] ta_mig invalid cells (==0): {int(bad.sum())}")
            arr = arr.astype(float)
            arr[bad] = np.nan

        if file_is_bottom_to_top:
            arr = arr[:, ::-1, :]

        print("[V2] after mask/flip:",
              "nan:", int(np.isnan(arr).sum()),
              "finite:", int(np.isfinite(arr).sum()),
              "min:", float(np.nanmin(arr)), "max:", float(np.nanmax(arr)))

        padded = np.full((arr.shape[0], n_src, n_cells), np.nan)
        padded[:, -n_file_lev:] = arr
        padded[:, : n_src - n_file_lev] = 0.0

        print("[P1] padded nan:", int(np.isnan(padded).sum()),
              "finite:", int(np.isfinite(padded).sum()),
              "top-level finite (level 0) count:", int(np.isfinite(padded[:, 0, :]).sum()),
              "bottom-level finite (last level) count:", int(np.isfinite(padded[:, -1, :]).sum()))

        out = np.empty((arr.shape[0], n_tgt, n_cells))

        for t in range(arr.shape[0]):
            out[t] = coarse_grain_vectorized(
                padded[t],
                source_top,
                source_bot,
                target_top,
                target_bot,
                mask_ref=None,       
                fill_value=0.0,     
            )

        ds_out[v] = (("time", "height", "cell"), out)
        ds_out[v].attrs = ds[v].attrs

        # quick check for ta_mig to ensure it's not wiped
        if v == "ta_mig":
            ta0 = out[0]
            print("[TA-OUT] time0 min/max:", float(np.nanmin(ta0)), float(np.nanmax(ta0)),
                  "==0 frac:", float(np.mean(ta0 == 0.0)))

    ds_out["z_top"] = (("height", "cell"), target_top)
    ds_out["z_bot"] = (("height", "cell"), target_bot)
    ds_out.attrs = ds.attrs

    outname = os.path.basename(input_file).replace("_processed.nc", f"_{n_tgt}levels.nc")
    outpath = os.path.join(output_dir, outname)
    print(f"\n→ Writing {outpath}")
    ds_out.to_netcdf(outpath)

    ds.close()
    grid_ds.close()
    tgt_ds.close()


# Driver
def main():
    output_dir = (
        "..."
        "coarse-grained-data/icon_a_ml_coarse_grained/48_levels"
    )
    os.makedirs(output_dir, exist_ok=True)

    input_dir = (
        "..."
        "coarse-grained-data/icon_a_ml_coarse_grained"
    )

    files = glob.glob(os.path.join(input_dir, "*0*T000000Z_R02B05_processed.nc"))

    grid_file = "vgrid_regridded.nc"
    target_grid_file = (
        "..."
        "experiments/tuned_mig_10_year/"
        "tuned_mig_10_year_vgrid_ml.nc"
    )

    for f in sorted(files):
        process_file(f, output_dir, grid_file, target_grid_file)


if __name__ == "__main__":
    main()
