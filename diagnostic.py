import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE DATA QUALITY ANALYSIS")
print("=" * 80)

# Load all splits
train_df = pd.read_parquet("output/data/train.parquet")
val_df = pd.read_parquet("output/data/val.parquet")
test_df = pd.read_parquet("output/data/test.parquet")

print(f"\n[1] DATASET SIZES")
print(f"Train: {len(train_df):,} sequences")
print(f"Val:   {len(val_df):,} sequences")
print(f"Test:  {len(test_df):,} sequences")

# Check 1: Target Distribution (CRITICAL!)
print(f"\n[2] TARGET DISTRIBUTION ANALYSIS")
train_targets = train_df['target'].tolist()
val_targets = val_df['target'].tolist()

target_counter = Counter(train_targets)
print(f"Unique songs in train targets: {len(target_counter):,}")

# Check if distribution is balanced or skewed
top_10_songs = target_counter.most_common(10)
print(f"\nTop 10 most common targets in training:")
for idx, (song, count) in enumerate(top_10_songs, 1):
    pct = count / len(train_targets) * 100
    print(f"  {idx}. Song (URI ending): ...{song[-20:]} - {count:,} times ({pct:.2f}%)")

# Calculate concentration
top_1_pct = top_10_songs[0][1] / len(train_targets) * 100
top_10_pct = sum(count for _, count in top_10_songs) / len(train_targets) * 100
top_100_pct = sum(count for _, count in target_counter.most_common(100)) / len(train_targets) * 100

print(f"\n🎯 Concentration Analysis:")
print(f"  Top 1 song:     {top_1_pct:.2f}% of all targets")
print(f"  Top 10 songs:   {top_10_pct:.2f}% of all targets")
print(f"  Top 100 songs:  {top_100_pct:.2f}% of all targets")

if top_1_pct > 5:
    print(f"  ⚠️  WARNING: Top song appears {top_1_pct:.1f}% - highly imbalanced!")
if top_10_pct > 30:
    print(f"  ⚠️  WARNING: Top 10 songs dominate {top_10_pct:.1f}% - very skewed!")
if top_100_pct > 60:
    print(f"  ⚠️  WARNING: Top 100 songs are {top_100_pct:.1f}% - long tail problem!")

# Check 2: Long tail - how many rare songs?
print(f"\n[3] LONG TAIL ANALYSIS")
rare_songs = sum(1 for count in target_counter.values() if count < 10)
medium_songs = sum(1 for count in target_counter.values() if 10 <= count < 100)
common_songs = sum(1 for count in target_counter.values() if count >= 100)

print(f"Song frequency distribution:")
print(f"  Rare (<10 times):     {rare_songs:,} songs ({rare_songs/len(target_counter)*100:.1f}%)")
print(f"  Medium (10-99):       {medium_songs:,} songs ({medium_songs/len(target_counter)*100:.1f}%)")
print(f"  Common (100+):        {common_songs:,} songs ({common_songs/len(target_counter)*100:.1f}%)")

if rare_songs / len(target_counter) > 0.5:
    print(f"  ⚠️  WARNING: {rare_songs/len(target_counter)*100:.1f}% of songs appear <10 times - not enough to learn!")

# Check 3: Vocabulary overlap between train and val
print(f"\n[4] VOCABULARY OVERLAP")
train_vocab = set(train_targets)
val_vocab = set(val_targets)

overlap = train_vocab & val_vocab
val_only = val_vocab - train_vocab

print(f"Songs in train: {len(train_vocab):,}")
print(f"Songs in val:   {len(val_vocab):,}")
print(f"Overlap:        {len(overlap):,} ({len(overlap)/len(val_vocab)*100:.1f}% of val)")
print(f"Val-only songs: {len(val_only):,} ({len(val_only)/len(val_vocab)*100:.1f}%)")

if len(val_only) / len(val_vocab) > 0.1:
    print(f"  ⚠️  WARNING: {len(val_only)/len(val_vocab)*100:.1f}% of val songs never seen in training - cold start issue!")

# Check 4: History diversity
print(f"\n[5] HISTORY DIVERSITY")

# Parse history
def get_history(row):
    if isinstance(row['history'], str):
        return row['history'].split('|') if row['history'] else []
    return row['history']

# Sample 1000 sequences
sample_size = min(1000, len(train_df))
sample_histories = [get_history(train_df.iloc[i]) for i in range(sample_size)]

# Check unique songs per position
position_diversity = {}
for history in sample_histories:
    for pos, song in enumerate(history):
        if pos not in position_diversity:
            position_diversity[pos] = set()
        position_diversity[pos].add(song)

print(f"Unique songs at each position (first 10 positions, from {sample_size} samples):")
for pos in range(min(10, len(position_diversity))):
    if pos in position_diversity:
        print(f"  Position {pos}: {len(position_diversity[pos])} unique songs")

# Check 5: Sequence patterns
print(f"\n[6] SEQUENCE PATTERN ANALYSIS")

# Check for repetitive sequences
def history_to_str(row):
    hist = get_history(row)
    return '|'.join(hist[-5:]) if len(hist) >= 5 else '|'.join(hist)

history_patterns = [history_to_str(train_df.iloc[i]) for i in range(min(10000, len(train_df)))]
pattern_counter = Counter(history_patterns)

duplicate_patterns = sum(1 for count in pattern_counter.values() if count > 1)
print(f"Checked {len(history_patterns):,} sequences")
print(f"Unique patterns: {len(pattern_counter):,}")
print(f"Duplicate patterns: {duplicate_patterns:,} ({duplicate_patterns/len(pattern_counter)*100:.1f}%)")

if duplicate_patterns / len(pattern_counter) > 0.3:
    print(f"  ⚠️  WARNING: {duplicate_patterns/len(pattern_counter)*100:.1f}% duplicates - model may memorize!")

# Check 6: Playlist diversity
print(f"\n[7] PLAYLIST DIVERSITY")
unique_playlists_train = train_df['playlist_id'].nunique()
unique_playlists_val = val_df['playlist_id'].nunique()

avg_seq_per_playlist = len(train_df) / unique_playlists_train

print(f"Unique playlists in train: {unique_playlists_train:,}")
print(f"Unique playlists in val:   {unique_playlists_val:,}")
print(f"Avg sequences per playlist: {avg_seq_per_playlist:.1f}")

if avg_seq_per_playlist < 3:
    print(f"  ⚠️  WARNING: Only {avg_seq_per_playlist:.1f} sequences per playlist - might not be enough context!")

# Check 7: History length distribution
print(f"\n[8] HISTORY LENGTH QUALITY")
history_lengths = [len(get_history(train_df.iloc[i])) for i in range(min(10000, len(train_df)))]

print(f"History length distribution (from 10K samples):")
print(f"  0-2 songs:   {sum(1 for l in history_lengths if l <= 2):,} ({sum(1 for l in history_lengths if l <= 2)/len(history_lengths)*100:.1f}%)")
print(f"  3-5 songs:   {sum(1 for l in history_lengths if 3 <= l <= 5):,} ({sum(1 for l in history_lengths if 3 <= l <= 5)/len(history_lengths)*100:.1f}%)")
print(f"  6-10 songs:  {sum(1 for l in history_lengths if 6 <= l <= 10):,}")
print(f"  11-20 songs: {sum(1 for l in history_lengths if 11 <= l <= 20):,}")
print(f"  20+ songs:   {sum(1 for l in history_lengths if l > 20):,}")

very_short = sum(1 for l in history_lengths if l <= 2) / len(history_lengths)
if very_short > 0.3:
    print(f"  ⚠️  WARNING: {very_short*100:.1f}% have ≤2 songs - hard to predict patterns!")

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)