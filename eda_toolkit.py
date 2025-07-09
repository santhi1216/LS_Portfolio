# ============================================
# Exploratory Data Analysis (EDA) Toolkit
# ============================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

# ------------------------------
# Data Cleaning
# ------------------------------
def clean(df, drop_duplicates=True, fix_columns=True):
    if fix_columns:
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        print(f"\u2705 Column names cleaned.")

    if drop_duplicates:
        num_duplicates = df.duplicated().sum()
        if num_duplicates == 0:
            print(f"\u2705 No duplicates found.")
        else:
            df = df.drop_duplicates()
            print(f"\u2705 {num_duplicates} duplicate rows removed.")
    return df


# ------------------------------
# Statistical Summary with Insights
# ------------------------------
def descriptive_insights_summary(df, highlight_missing=True, include_skew_kurtosis=True):
    desc = df.describe()
    display(desc)

    print(f"\nThere are total {len(df)} records in the dataset.\n")

    for col in desc.columns:
        mean = desc[col]['mean']
        median = desc[col]['50%']
        std = desc[col]['std']
        min_val = desc[col]['min']
        max_val = desc[col]['max']
        skew = df[col].skew()
        missing_count = df[col].isnull().sum()
        missing_percent = 100 * missing_count / len(df)

        if abs(mean - median) < (0.1 * abs(mean)):
            skewness_desc = "The data distribution is symmetric."
        elif mean > median:
            skewness_desc = "The data distribution is moderately right-skewed."
        else:
            skewness_desc = "The data distribution is moderately left-skewed."

        print(f"- '{col}' ranges from {min_val} to {max_val} with an average of {round(mean, 2)}.")
        print(f"  → {skewness_desc}")

        if highlight_missing and missing_count > 0:
            print(f"  → \033[1m⚠️ Missing values: {missing_count} ({round(missing_percent, 2)}%). Recommend imputation.\033[0m")

        print()


# ------------------------------
# Outliers Analysis
# ------------------------------
def outlier_analysis(df, num_cols, iqr_factor=1.5):
    outlier_summary = []
    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        perc = (outliers / len(data)) * 100
        outlier_summary.append({'Column': col, 'Outliers': outliers, 'Percentage': round(perc, 2)})
    summary_df = pd.DataFrame(outlier_summary).sort_values(by='Percentage', ascending=False).reset_index(drop=True)
    print("\n\ud83d\udccc **Outlier Summary (Based on IQR method):**\n")
    print(summary_df)
    return summary_df


# ------------------------------
# Handling Outliers
# ------------------------------
def cap_outliers_iqr(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df[column] = np.clip(df[column], lower, upper)


# ------------------------------
# Univariate Analysis - Categorical
# ------------------------------
def univariate_analysis_categorical(df, cat_cols, sample_size=5000, bins=30, top_n=10, max_unique=50):
    print("\n===== Categorical Features =====\n")

    if df.shape[0] > sample_size:
        df = df.sample(sample_size, random_state=42)

    n_cols = 3
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    obs_cat = []

    for i, col in enumerate(cat_cols):
        print(f"Processing: {col}")
        if df[col].nunique() > max_unique:
            print(f"\u26a0\ufe0f Skipping '{col}' — too many unique values ({df[col].nunique()}).")
            fig.delaxes(axes[i])
            continue

        top_cats = df[col].value_counts(normalize=True).head(top_n)
        sns.barplot(x=top_cats.index, y=top_cats.values, palette='viridis', ax=axes[i])
        axes[i].set_title(f"{col}", fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Proportion')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)


# ------------------------------
# Univariate Analysis - Numerical
# ------------------------------
def plot_numerical_distributions(df, numerical_columns, bins=30, figsize=(6, 4), max_cols=3):
    total = len(numerical_columns)
    ncols = max_cols
    nrows = (total + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_columns):
        sns.histplot(data=df, x=col, bins=bins, kde=False, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# ------------------------------
# Bivariate Analysis
# ------------------------------
def bivariate_default_rate_summary(df, feature, target, bins=None, qcut=False, quantiles=4, positive_label=None):
    data = df.copy()

    if positive_label is None:
        positive_label = data[target].value_counts().idxmax()

    if pd.api.types.is_numeric_dtype(data[feature]):
        if qcut:
            data[feature + '_binned'] = pd.qcut(data[feature], q=quantiles, duplicates='drop')
        elif bins:
            data[feature + '_binned'] = pd.cut(data[feature], bins=bins)
        else:
            data[feature + '_binned'] = pd.cut(data[feature], bins=5)
        group_col = feature + '_binned'
    else:
        group_col = feature

    pivot = data.groupby(group_col)[target].value_counts(normalize=True).unstack().fillna(0)

    if positive_label not in pivot.columns:
        print(f"\u26a0\ufe0f '{positive_label}' not found in target column '{target}'")
        return

    pivot['event_rate'] = pivot[positive_label] * 100
    pivot_sorted = pivot[['event_rate']].sort_values(by='event_rate', ascending=False)

    print(f"\n\ud83d\udccc Event Rate by '{feature}':")
    print(pivot_sorted)

    max_bin = pivot_sorted.index[0]
    max_rate = pivot_sorted.iloc[0]['event_rate']
    min_bin = pivot_sorted.index[-1]
    min_rate = pivot_sorted.iloc[-1]['event_rate']
    delta = max_rate - min_rate

    print(f"\n\ud83d\udd0d Group '{max_bin}' has the highest event rate: {max_rate:.2f}%.")
    print(f"\ud83d\udfe2 Lowest event rate is in group '{min_bin}': {min_rate:.2f}%.")
    print(f"\ud83d\udcca Spread between highest and lowest: {delta:.2f}%.\n")

    pivot['event_rate'].plot(kind='bar', color='steelblue', figsize=(9, 4))
    plt.ylabel('Event Rate (%)')
    plt.title(f'Event Rate by {feature}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
