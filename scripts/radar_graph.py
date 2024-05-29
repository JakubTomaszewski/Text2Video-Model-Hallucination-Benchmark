import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from textwrap import wrap
import seaborn as sns

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="evaluation_res", help="Path to the folder with the resutls.")
    return parser.parse_args()


def merge_and_sum_multiple(dfs, on):
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=on, suffixes=("", "_drop"))
        
        # Find columns to sum and drop duplicates
        for col in df.columns:
            if col in on:
                continue
            if f"{col}_drop" in merged.columns:
                merged[col] += merged[f"{col}_drop"]
                merged.drop(columns=[f"{col}_drop"], inplace=True)
    
    for col in merged.columns:
        if col not in on:
            merged[col] = merged[col] / len(dfs)
    return merged

def concatenate_color_res():
    path1 = "color_res"
    total = []
    for file in os.listdir(path1):
        if file.endswith(".csv"):
            data = pd.read_csv(f"{path1}/{file}")
            total.append(data)

    total_data = merge_and_sum_multiple(total, on=["model_name","video"])
    total_data.to_csv("evaluation_res/total_color_res.csv", index=False)

def plot_radar_chart(data, title):
    column_names = {'object_score':'Object Detection', 'count_score': 'Object Counting', 'color':'Color Identification',
       'confusion': 'Color Attribution', 'stage1': 'Video Captioning'}
    names = {'7b': 'Text-to-video-ms-1.7b', 
    'AnimateDiff_Lightning': 'AnimateDiff-Lightning', 
    'Animatediff_motion_adapter':'Animatediff-motion-adapter-v1-5',
    'potat1':'Potat1',
    'sd_v1_5':'Stable-diffusion-v1-5',
    'sd_xl': 'Stable-diffusion-xl-base-1.0',
    'zeroscope': 'Zeroscope_v2_576w'}

    metrics = list(column_names.keys())
    N = len(metrics)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.spines["polar"].set_zorder(1)
    ax.spines["polar"].set_color("lightgrey")

    color_palette = ["#FF9900", "#8354ad", "#b9e2f5",  "#77ab59", "crimson", "#0500FF", "thistle"]
    color_idx = 0
    for idx in range(len(data)):
        values = data.iloc[idx, 2:].values * 100
        values = values.tolist()
        values = values + [values[0]]
        ax.plot(theta, values, linewidth=2.75, linestyle="solid", label=names[data.iloc[idx, 0]], marker="o", markersize=1, color=color_palette[color_idx])
        ax.fill(theta, values, alpha=0.50, color=color_palette[color_idx])
        color_idx += 1

    # Set the labels
    REGION = ["\n".join(wrap(r, 10, break_long_words=False)) for r in metrics]
    ax.set_xticks(theta)
    ax.set_xticklabels(REGION  + [REGION[0]], size=15)

    plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], color="black", size=12)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), prop={'size': 15}) 
    plt.tight_layout()
    plt.savefig(f"{title}.png")

if __name__=="__main__":
    # Aggregate the results from the color task (cogvml+internvl)
    concatenate_color_res()

    args = argument_parser()
    total_data = pd.DataFrame()
    for i, file in enumerate(os.listdir(args.input_folder)):
        if file.endswith(".csv"):
            data = pd.read_csv(f"{args.input_folder}/{file}")
            if i==0:
                total_data = data
            else:
                total_data = total_data.merge(data, on=["model_name","video"], how="right")    

    grouped = total_data.groupby("video")
    for group in grouped.groups:
        group_data = grouped.get_group(group)
        plot_radar_chart(group_data, f"Evaluation for videos with {int(group)}secs duration")

