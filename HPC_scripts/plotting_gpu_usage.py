# Plotting the GPU utilization and  memory used over time, output by nvidia-smi:
#    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv --loop=1 >> gpu_monitor_h100_16Feb26.log

from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--logfile", dest="logfile", default='', type=str, help="nvidia-smi output")
    parser.add_argument("--output", dest="outputfile", default='gpu_usage.pdf', type=str, help="PDF output")
    parser.add_argument("--cutoff", dest="cutoff", default='', type=str, help="Time cutoff, e.g. 2026/02/16 08:43:39.177 ")

    args = parser.parse_args()
    filename = args.logfile
    outputfile = args.outputfile
    cutoff = None
    if args.cutoff != '':
        cutoff = pd.to_datetime(args.cutoff)


    # Read raw file
    # Load file (with header)
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()

    # Rename columns to something sane
    df = df.rename(columns={
        "index": "gpu",
        "utilization.gpu [%]": "util",
        "memory.used [MiB]": "mem_used",
        "memory.total [MiB]": "mem_total",
    })

    # Clean columns
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["util"] = df["util"].astype(str).str.replace("%", "").astype(float)
    df["mem_used"] = df["mem_used"].str.replace("MiB", "", regex=False).astype(float)

    # Plot utilization for each GPU
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot per GPU
    for gpu_id in sorted(df["gpu"].unique()):
        g = df[df["gpu"] == gpu_id]
        ax1.plot(g["timestamp"], g["util"], label=f"GPU {gpu_id}")
        ax2.plot(g["timestamp"], g["mem_used"], label=f"GPU {gpu_id}")

    ax1.set_ylabel("Utilization (%)")
    ax1.set_title("GPU Utilization Over Time")
    

    ax2.set_ylabel("Memory Used (MiB)")
    ax2.set_xlabel("Time")
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if cutoff is not None:
        ax1.set_xlim(right=cutoff)
        ax2.set_xlim(right=cutoff)

    ax1.legend(ncol=2, fontsize=8)

    fig.autofmt_xdate()
    plt.tight_layout()

    # Save to PDF
    plt.savefig(outputfile)
    plt.show()