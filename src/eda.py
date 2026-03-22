import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected column 'Class' not found.")
    return df

def plot_class_balance(df, out_path):
    counts = df["Class"].value_counts().sort_index()
    plt.figure(figsize=(5,4))
    ax = sns.barplot(x=counts.index.map({0:"Non-Fraud",1:"Fraud"}), y=counts.values, palette=["#4C72B0","#DD8452"])
    for p,v in zip(ax.patches, counts.values):
        ax.annotate(f"{v:,}", (p.get_x()+p.get_width()/2, p.get_height()), ha="center", va="bottom")
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    ratio = counts.get(0,0) / max(1, counts.get(1,1))
    return counts.to_dict(), ratio

def plot_amount_hist(df, out_path):
    plt.figure(figsize=(6,4))
    log_amt = np.log1p(df["Amount"].values)
    sns.histplot(log_amt, bins=60, kde=True, color="#4C72B0")
    plt.title("Amount (log1p) Distribution")
    plt.xlabel("log1p(Amount)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_time_hist(df, out_path):
    plt.figure(figsize=(6,4))
    sns.histplot(df["Time"], bins=60, kde=False, color="#55A868")
    plt.title("Time Distribution")
    plt.xlabel("Time (seconds from first transaction)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_corr_heatmap(df, out_path, max_features=30):
    num_df = df.select_dtypes(include=[np.number])
    cols = list(num_df.columns)
    priority = [c for c in ["Class","Amount","Time"] if c in cols]
    vcols = [c for c in cols if c not in priority]
    keep = priority + vcols[:max_features - len(priority)]
    corr = num_df[keep].corr()
    plt.figure(figsize=(min(0.45*len(keep)+2, 14), min(0.45*len(keep)+2, 14)))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False, cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    if "Class" in corr.columns:
        class_corr = corr["Class"].drop("Class", errors="ignore").abs().sort_values(ascending=False)
        return class_corr.to_dict()
    return {}

def print_key_stats(df):
    total = len(df)
    pos = int((df["Class"]==1).sum())
    neg = total - pos
    pos_ratio = pos/total
    print(f"Total samples: {total:,}")
    print(f"Fraud: {pos:,} ({pos_ratio:.4%}), Non-Fraud: {neg:,}")
    # Amount stats
    amt = df["Amount"]
    print("\nAmount stats (overall):")
    print(amt.describe(percentiles=[0.5,0.9,0.99]).to_string())
    # Amount by class
    fraud_amt = df.loc[df["Class"]==1, "Amount"]
    non_amt = df.loc[df["Class"]==0, "Amount"]
    def q(s): 
        return {"median": float(s.median()), "p90": float(s.quantile(0.9)), "p99": float(s.quantile(0.99))}
    print("\nAmount by class (median, p90, p99):")
    print(f"  Fraud: {q(fraud_amt)}")
    print(f"  Non-Fraud: {q(non_amt)}")
    if "Time" in df.columns:
        print("\nTime stats (overall):")
        print(df["Time"].describe(percentiles=[0.25,0.5,0.75]).to_string())

def main(args):
    ensure_dir("figures")
    df = load_data(args.data_path)

    # 1) Class balance
    counts, ratio = plot_class_balance(df, "figures/class_balance.png")
    # 2) Amount distribution (log1p)
    plot_amount_hist(df, "figures/amount_hist.png")
    # 3) Time distribution
    if "Time" in df.columns:
        plot_time_hist(df, "figures/time_hist.png")
    # 4) Correlation heatmap
    class_corr = plot_corr_heatmap(df, "figures/corr_heatmap.png", max_features=30)

    # Print summary to console
    print_key_stats(df)
    print("\nClass counts:", counts)
    print(f"Negative/Positive ratio ≈ {ratio:.1f}:1")
    if class_corr:
        top10 = sorted(class_corr.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop-10 absolute correlations with Class:")
        for k,v in top10:
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    args = p.parse_args()
    main(args)
