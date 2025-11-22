import pandas as pd
from pathlib import Path

# Load the sequences you already created
sequences_df = pd.read_parquet("output/data/train.parquet")

print("="*80)
print("VOCABULARY CHECK")
print("="*80)

# Vocabulary size
unique_targets = sequences_df['target'].nunique()
print(f"\n🎯 VOCABULARY SIZE: {unique_targets:,} unique songs")

# History length analysis
if 'history_str' in sequences_df.columns:
    # If you saved history as string
    history_lengths = sequences_df['history_str'].apply(lambda x: len(x.split('|')) if x else 0)
else:
    # If you saved history as list
    history_lengths = sequences_df['history'].apply(len)

print(f"\nHistory Length Statistics:")
print(f"  Mean: {history_lengths.mean():.1f} songs")
print(f"  Median: {history_lengths.median():.0f} songs")
print(f"  Min: {history_lengths.min()}")
print(f"  Max: {history_lengths.max()}")

print(f"\nTotal sequences: {len(sequences_df):,}")

# Distribution
print(f"\nHistory Length Distribution:")
for lower, upper in [(1, 5), (6, 10), (11, 20), (21, 50)]:
    count = ((history_lengths >= lower) & (history_lengths <= upper)).sum()
    pct = count / len(history_lengths) * 100
    print(f"  {lower}-{upper} songs: {count:,} ({pct:.1f}%)")

count_50plus = (history_lengths > 50).sum()
pct_50plus = count_50plus / len(history_lengths) * 100
print(f"  50+ songs: {count_50plus:,} ({pct_50plus:.1f}%)")

print("\n" + "="*80)