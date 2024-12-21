import os
import argparse
from tabnanny import verbose

import numpy as np
import pandas as pd
import pyBigWig
import re
import pyranges
from numba import njit
from MFDFA import MFDFA
from scipy.constants import value
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.stattools import acf
from matplotlib import pyplot as plt

@njit
def _compute_fluctuation(profile, scale):
    """
    Compute the fluctuation (RMS of detrended data) for a given scale.
    """
    n = len(profile)
    segments = n // scale
    rms = np.zeros(segments)
    for i in range(segments):
        segment = profile[i * scale:(i + 1) * scale]
        # Linear detrend using a polynomial fit (a*x + b)
        x = np.arange(scale)
        mx = (scale - 1) / 2.0
        my = np.mean(segment)
        cov_xy = np.sum((np.arange(scale) - mx) * (segment - my))
        var_x = np.sum((np.arange(scale) - mx) ** 2)
        a = cov_xy / var_x
        b = my - a * mx
        detrended = segment - (a * np.arange(scale) + b)
        rms[i] = np.sqrt(np.mean(detrended ** 2))
    return np.mean(rms)

def dfa(signal, scales):
    # Step 1: Integrate the signal (profile)
    signal = np.asarray(signal)
    signal = signal - np.mean(signal)
    profile = np.cumsum(signal)

    # Step 2: Compute fluctuation for each scale
    fluct = np.zeros(len(scales))
    for i, s in enumerate(scales):
        fluct[i] = _compute_fluctuation(profile, s)

    # Step 3: Fit line to log-log
    log_scales = np.log(scales)
    log_fluct = np.log(fluct)
    slope, intercept = np.polyfit(log_scales, log_fluct, 1)
    return slope

def compute_base_stat(values:np.array):
    out_stat = {}
    out_stat["mean"] = np.nanmean(values)
    for i in range(1,30):
        out_stat[f"q{i}"] = np.nanquantile(values, i/30)
    out_stat["std"] = np.nanstd(values)
    out_stat["min"] = np.nanmin(values)
    out_stat["max"] = np.nanmax(values)
    out_stat["sum"] = np.nansum(values)
    out_stat["count"] = np.sum(~np.isnan(values))
    out_stat["coverage"] = out_stat["count"] / len(values)
    out_stat["nans"] = np.sum(np.isnan(values))
    out_stat["zeros"] = np.sum(values == 0)
    out_stat["above_q7"] = np.sum(values > out_stat["q7"])/len(values)
    out_stat["below_q3"] = np.sum(values < out_stat["q3"])/len(values)
    out_stat["between_q3_q7"] = np.sum((values >= out_stat["q3"]) & (values <= out_stat["q7"]))/len(values)
    return out_stat

def parse_filename(filename):
    pattern = r"^(\d{5})([NT])_?([a-zA-Z0-9_]+)\.bw$"
    match = re.match(pattern, filename)
    if match:
        sample_id, tissue_type, antibody_id = match.groups()
        antibody_id = re.sub(r"_R.*", "", antibody_id)
        if antibody_id in ["DeltaNp63", "Delta_N_p63"]:
            antibody_id = "Delta_N_p63"
        return {"sample_id": sample_id, "N/T": tissue_type, "antibody_id": antibody_id}
    return None

def process_bw_file(filepath, chrom, start, end,genes,genes_as_features = False, verbose = 0):
    filename = os.path.basename(filepath)
    out_dict = {}
    out_dict["filename"] = filename
    try:
        bw = pyBigWig.open(filepath)
        if not bw.chroms():
            if verbose > 1:
                print(f"Skipping '{filename}': No chromosomes found in BigWig file.")
            bw.close()
            return None

        # If chrom, start, or end not provided, pick defaults
        values = []
        if genes is None:
            values = bw.values(chrom, start, end, numpy=True)
            if verbose > 1:
                print(f"Processing {filename} for {chrom}:{start}-{end} - number of values {len(values)}")

        else:
            for index,gene in genes.df.iterrows():
                gene_start = gene['Start']
                gene_end = gene['End']
                gene_chrom = gene['Chromosome']
                gene_values = bw.values(gene_chrom, gene_start, gene_end, numpy=True)
                if not genes_as_features:
                    values.extend(gene_values)
                else:
                    values.append(gene_values.tolist())
                    raise("Not implemented")
            if verbose > 1:
                if genes_as_features:
                    print(f"Processing {filename} for {len(values)} fragments")
                else:
                    print(f"Processing {filename} for {len(values)}values from custom set of genes")
            if not genes_as_features:
                values = np.array(values)
                values = np.nan_to_num(values)
        #replace NaNs with zeros

        bw.close()

        out_stat = compute_base_stat(values)
        file_attrs = parse_filename(filename)
        out_dict.update(out_stat)
        if file_attrs is not None:
            out_dict.update(file_attrs)

        scales = np.unique(np.logspace(0.1, 4, 20).astype(int))
        for q in [1,2,3,4,5,6]:
            lag, dfar = MFDFA(values, q=q, order=1, lag=scales)
            # polyfit returns (slope, intercept), slope is index 0
            out_dict[f"MFDFA{q}"] =  np.polyfit(np.log(lag)[4:], np.log(dfar[4:]), 1)[0][0]

        acf_data = acf(values, nlags=3000)
        #keep each 100th value
        acf_data_100 = acf_data[::100]
        i = 0
        for v in acf_data_100:
            out_dict[f"ACF_{i*100}"] = v
            i = i + 1
        return out_dict
    except Exception as e:
        print(f"Error processing '{filename}': {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create table with statistics of bigwig files")
    parser.add_argument("-indir", type=str, help="Path to the directory containing BigWig files.", required=True)
    parser.add_argument("--chr", type=str, help="Chromosome name (e.g., 'chr12').", default=None)
    parser.add_argument("--start", type=int, help="Start position of the interval (0-based).", default=None)
    parser.add_argument("--end", type=int, help="End position of the interval.", default=None)
    parser.add_argument("--genes", type=str, help="Path to bed file with regions for analysis", default=None)
    parser.add_argument("--genes_as_features", help="If yes all features will be computed for each gene separetely", default=False,
                        action='store_true')

    parser.add_argument("--outcsv", type=str, help="Path to output CSV", default="stats.csv")
    parser.add_argument("--verbose", type=int, help="Verbose level", default=2)
    parser.add_argument("--workers", type=int, help="Number of parallel workers", default=None)

    args = parser.parse_args()

    directory = args.indir
    chrom = args.chr
    start = args.start
    end = args.end
    genesfile = args.genes
    genes_as_features = args.genes_as_features

    #read genes from comma separated file
    if genesfile is not None:
        genes = pyranges.read_bed(genesfile)
        if args.verbose > 0:
            print(f"Loaded {len(genes)} regions from {genesfile}")
        #merge all regions
        genes = genes.merge(strand=False)
        if args.verbose > 0:
            print(f"After merge remaining {len(genes)} regions")
        #sort by chromosome and start
        genes = genes.sort()
        if args.verbose > 2:
            print(f"Regions for features estimation:")
            for i,r in genes.df.iterrows():
                print(f"{r['Chromosome']} {r['Start']} {r['End']}")
    else:
        genes = None
        if chrom is None or start is None or end is None:
            print("Error: If genes are not provided, chrom, start, and end must be specified.")
            exit(1)

    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")


    # Collect all bigwig files
    bw_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".bw")]
    stat_list = []
    # If verbose==1, we want to show tqdm progress
    # If verbose>1, we'll print info inside the worker functions themselves
    show_progress = (args.verbose == 1)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_bw_file, fp, chrom, start, end, genes,genes_as_features, args.verbose) for fp in bw_files]

        if show_progress:
            # Using as_completed with tqdm
            for fut in tqdm(as_completed(futures), total=len(futures)):
                result = fut.result()
                if result is not None:
                    stat_list.append(result)
        else:
            # No progress bar
            for fut in futures:
                result = fut.result()
                if result is not None:
                    stat_list.append(result)

    df = pd.DataFrame(stat_list)
    df.to_csv(args.outcsv, index=False)
    if args.verbose > 0:
        print(f"Saved results to {args.outcsv}")
