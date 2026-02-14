#!/usr/bin/env python3
"""
backfill_metrics.py - Parse existing training logs and create metrics CSV files.

Parses:
  - logs/tournament.log  (tournament rounds + extended training)
  - logs/training.log    (resumed training with new script format)

Outputs:
  - logs/metrics.csv           (ongoing training data for charts)
  - logs/tournament_metrics.csv (tournament round data)
  - logs/samples.jsonl          (sample generation events)
"""

import csv
import json
import math
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

TOURNAMENT_LOG = os.path.join(LOGS_DIR, "tournament.log")
TRAINING_LOG = os.path.join(LOGS_DIR, "training.log")

METRICS_CSV = os.path.join(LOGS_DIR, "metrics.csv")
TOURNAMENT_CSV = os.path.join(LOGS_DIR, "tournament_metrics.csv")
SAMPLES_JSONL = os.path.join(LOGS_DIR, "samples.jsonl")

# Tournament used batch_size=16, seq_len=512
TOURNAMENT_BATCH_SIZE = 16
TOURNAMENT_SEQ_LEN = 512
TOURNAMENT_TOKENS_PER_STEP = TOURNAMENT_BATCH_SIZE * TOURNAMENT_SEQ_LEN  # 8192

# Extended training in tournament.log used batch_size=16, seq_len=1024
EXTENDED_BATCH_SIZE = 16
EXTENDED_SEQ_LEN = 1024
EXTENDED_TOKENS_PER_STEP = EXTENDED_BATCH_SIZE * EXTENDED_SEQ_LEN  # 16384

# New training script also uses batch_size=16, seq_len=1024
TRAINING_BATCH_SIZE = 16
TRAINING_SEQ_LEN = 1024
TRAINING_TOKENS_PER_STEP = TRAINING_BATCH_SIZE * TRAINING_SEQ_LEN  # 16384


def parse_tournament_log():
    """Parse tournament.log for tournament rounds and extended training data."""
    tournament_rows = []
    extended_rows = []

    # Patterns for tournament round training lines
    # Format:  step N/M | loss=X | lr=X | N tok/s
    tournament_step_re = re.compile(
        r'^\s+step\s+(\d+)/(\d+)\s+\|\s+loss=([\d.]+)\s+\|\s+lr=([\d.eE+-]+)\s+\|\s+(\d+)\s+tok/s'
    )

    # Pattern for extended training lines (from tournament.log champion training)
    # Format: step N | loss=X | lr=X | N tok/s | X.Xh elapsed
    extended_step_re = re.compile(
        r'^\s+step\s+(\d+)\s+\|\s+loss=([\d.]+)\s+\|\s+lr=([\d.eE+-]+)\s+\|\s+(\d+)\s+tok/s\s+\|\s+([\d.]+)h\s+elapsed'
    )

    # Track current variant and round
    current_variant = None
    current_round = 0

    with open(TOURNAMENT_LOG, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # Detect round headers
            if 'ROUND 1:' in line:
                current_round = 1
                continue
            elif 'ROUND 2:' in line:
                current_round = 2
                continue
            elif 'ROUND 3:' in line:
                current_round = 3
                continue
            elif 'CHAMPION:' in line:
                current_round = 0  # Extended training section
                current_variant = None
                continue

            # Detect variant (Training or Continuing lines)
            m_train = re.match(r'^Training\s+(v\d+_\w+)\.\.\.', line)
            m_cont = re.match(r'^Continuing\s+(v\d+_\w+)\.\.\.', line)
            if m_train:
                current_variant = m_train.group(1)
                continue
            if m_cont:
                current_variant = m_cont.group(1)
                continue

            # Tournament step lines
            if current_round > 0 and current_variant:
                m = tournament_step_re.match(line)
                if m:
                    step = int(m.group(1))
                    max_step = int(m.group(2))
                    loss = float(m.group(3))
                    lr = m.group(4)
                    tok_s = int(m.group(5))

                    tournament_rows.append({
                        'step': step,
                        'variant': current_variant,
                        'round': current_round,
                        'loss': loss,
                        'learning_rate': lr,
                        'tokens_per_sec': tok_s,
                    })
                    continue

            # Extended training step lines (no variant, round=0)
            if current_round == 0:
                m = extended_step_re.match(line)
                if m:
                    step = int(m.group(1))
                    loss = float(m.group(2))
                    lr = m.group(3)
                    tok_s = int(m.group(4))
                    elapsed_h = float(m.group(5))

                    tokens_total = step * EXTENDED_TOKENS_PER_STEP
                    tokens_billions = tokens_total / 1e9
                    perplexity = math.exp(loss) if loss < 20 else float('inf')

                    extended_rows.append({
                        'timestamp': '',  # No timestamps in tournament.log extended section
                        'step': step,
                        'loss': loss,
                        'perplexity': round(perplexity, 1),
                        'learning_rate': lr,
                        'tokens_per_sec': tok_s,
                        'tokens_total': tokens_total,
                        'tokens_billions': round(tokens_billions, 2),
                        'elapsed_hours': elapsed_h,
                        'phase': 'extended_v1',
                    })
                    continue

    return tournament_rows, extended_rows


def parse_training_log():
    """Parse training.log for resumed training data and sample generations.

    The training log may contain multiple restarts (the training process was
    restarted a few times). We handle this by:
    1. Collecting all step data as we go
    2. When we detect a new "HAMNER PRE-TRAINING" header, we know a restart
       occurred. We keep only unique step data, preferring the LAST occurrence
       (from the final successful run) for any given step number.
    3. Similarly for samples, we deduplicate by step number.
    """
    # We'll collect ALL rows first, then deduplicate
    all_training_rows = []
    all_samples = []

    # Training step line pattern:
    # [2026-02-14 12:13:59] step   30550 | loss 5.9255 | ppl 374.5 | lr 2.94e-04 | 11712 tok/s | 0.50B tokens (10.2%) | 0.0h
    step_re = re.compile(
        r'^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+step\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+ppl\s+([\d.]+)\s+\|\s+lr\s+([\d.eE+-]+)\s+\|\s+(\d+)\s+tok/s\s+\|\s+([\d.]+)B\s+tokens\s+\(([\d.]+)%\)\s+\|\s+([\d.]+)h'
    )

    # Sample generation block detection
    sample_header_re = re.compile(
        r'^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+---\s+SAMPLE GENERATIONS\s+---'
    )
    sample_line_re = re.compile(
        r'^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]\s+\[(\d+)\]\s+(.*)'
    )
    sample_end_re = re.compile(
        r'^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s+-{20,}'
    )

    in_samples = False
    current_sample_timestamp = None
    current_samples = {}
    current_sample_num = None
    current_sample_text = None
    last_step = None
    last_tokens_total = None
    last_tokens_billions = None

    with open(TRAINING_LOG, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Check for training step
        m = step_re.match(line)
        if m:
            timestamp = m.group(1)
            step = int(m.group(2))
            loss = float(m.group(3))
            ppl = float(m.group(4))
            lr = m.group(5)
            tok_s = int(m.group(6))
            tokens_b = float(m.group(7))
            tokens_pct = float(m.group(8))
            elapsed_h = float(m.group(9))

            tokens_total = int(tokens_b * 1e9)
            tokens_billions = tokens_b

            all_training_rows.append({
                'timestamp': timestamp,
                'step': step,
                'loss': loss,
                'perplexity': ppl,
                'learning_rate': lr,
                'tokens_per_sec': tok_s,
                'tokens_total': tokens_total,
                'tokens_billions': tokens_billions,
                'elapsed_hours': elapsed_h,
                'phase': 'training_v2',
            })

            last_step = step
            last_tokens_total = tokens_total
            last_tokens_billions = tokens_billions
            i += 1
            continue

        # Check for sample generation header
        m = sample_header_re.match(line)
        if m:
            in_samples = True
            current_sample_timestamp = m.group(1)
            current_samples = {}
            current_sample_num = None
            current_sample_text = None
            i += 1
            continue

        # Check for sample end
        if in_samples and sample_end_re.match(line):
            # Save the last sample text
            if current_sample_num is not None and current_sample_text is not None:
                current_samples[str(current_sample_num)] = current_sample_text.strip()

            # Write the sample event
            if current_samples:
                all_samples.append({
                    'timestamp': current_sample_timestamp,
                    'step': last_step,
                    'tokens_total': last_tokens_total,
                    'tokens_billions': last_tokens_billions,
                    'samples': current_samples,
                })

            in_samples = False
            current_sample_num = None
            current_sample_text = None
            i += 1
            continue

        # Inside samples block - parse individual samples
        if in_samples:
            # Check for new numbered sample line
            m = sample_line_re.match(line)
            if m:
                # Save previous sample if any
                if current_sample_num is not None and current_sample_text is not None:
                    current_samples[str(current_sample_num)] = current_sample_text.strip()

                current_sample_num = int(m.group(2))
                current_sample_text = m.group(3)
                i += 1
                continue
            else:
                # Continuation of previous sample (multiline output)
                if current_sample_num is not None:
                    # Strip timestamp prefix if present
                    stripped = line
                    ts_prefix = re.match(r'^\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s*(.*)', line)
                    if ts_prefix:
                        stripped = ts_prefix.group(1)
                    # Only append non-empty continuation
                    if stripped.strip():
                        current_sample_text += '\n' + stripped
                i += 1
                continue

        i += 1

    # Deduplicate training rows: keep the LAST occurrence for each step
    # (the final successful run's data is what we want)
    seen_steps = {}
    for row in all_training_rows:
        seen_steps[row['step']] = row  # Later entries overwrite earlier ones

    training_rows = sorted(seen_steps.values(), key=lambda r: r['step'])

    # Deduplicate samples: keep the LAST occurrence for each step
    seen_sample_steps = {}
    for sample in all_samples:
        seen_sample_steps[sample['step']] = sample

    samples_list = sorted(seen_sample_steps.values(), key=lambda s: s['step'])

    return training_rows, samples_list


def combine_metrics(extended_rows, training_rows):
    """
    Combine extended training data (from tournament.log) with resumed training
    data (from training.log) into a single unified metrics.csv.

    The training.log picks up from where extended training left off. The new
    training script uses a higher learning rate (warm restart), so the data
    represents a distinct continuation phase.

    Strategy:
    - Use ALL extended_rows (steps 100 to ~30900)
    - Use training_rows ONLY for steps > max extended step
      (the overlap region represents the new script re-training those steps
       with different hyperparameters; we keep the original extended data
       as canonical for the overlap range, and start training_v2 after)
    """
    if not extended_rows:
        return training_rows

    max_extended_step = max(r['step'] for r in extended_rows)

    combined = list(extended_rows)
    for row in training_rows:
        if row['step'] > max_extended_step:
            combined.append(row)

    # Sort by step
    combined.sort(key=lambda r: r['step'])
    return combined


def write_tournament_csv(tournament_rows):
    """Write tournament_metrics.csv"""
    fieldnames = ['timestamp', 'step', 'variant', 'round', 'loss', 'learning_rate', 'tokens_per_sec']

    with open(TOURNAMENT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in tournament_rows:
            writer.writerow({
                'timestamp': '',  # Tournament log has no timestamps
                'step': row['step'],
                'variant': row['variant'],
                'round': row['round'],
                'loss': row['loss'],
                'learning_rate': row['learning_rate'],
                'tokens_per_sec': row['tokens_per_sec'],
            })


def write_metrics_csv(rows):
    """Write metrics.csv"""
    fieldnames = [
        'timestamp', 'step', 'loss', 'perplexity', 'learning_rate',
        'tokens_per_sec', 'tokens_total', 'tokens_billions', 'elapsed_hours', 'phase'
    ]

    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_samples_jsonl(samples_list):
    """Write samples.jsonl"""
    with open(SAMPLES_JSONL, 'w') as f:
        for sample in samples_list:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    print("=" * 60)
    print("  Backfill Metrics from Training Logs")
    print("=" * 60)

    # Parse tournament log
    print(f"\nParsing {TOURNAMENT_LOG}...")
    tournament_rows, extended_rows = parse_tournament_log()
    print(f"  Tournament training rows: {len(tournament_rows)}")

    # Count per round/variant
    variants_per_round = {}
    for row in tournament_rows:
        key = (row['round'], row['variant'])
        variants_per_round[key] = variants_per_round.get(key, 0) + 1
    rounds_summary = {}
    for (rnd, var), count in variants_per_round.items():
        if rnd not in rounds_summary:
            rounds_summary[rnd] = {'variants': set(), 'steps': 0}
        rounds_summary[rnd]['variants'].add(var)
        rounds_summary[rnd]['steps'] += count
    for rnd in sorted(rounds_summary):
        info = rounds_summary[rnd]
        print(f"    Round {rnd}: {len(info['variants'])} variants, {info['steps']} step entries")

    print(f"  Extended training rows: {len(extended_rows)}")
    if extended_rows:
        print(f"    Steps: {extended_rows[0]['step']} to {extended_rows[-1]['step']}")

    # Parse training log
    print(f"\nParsing {TRAINING_LOG}...")
    training_rows, samples_list = parse_training_log()
    print(f"  Training step rows: {len(training_rows)}")
    if training_rows:
        print(f"    Steps: {training_rows[0]['step']} to {training_rows[-1]['step']}")
    print(f"  Sample generation events: {len(samples_list)}")

    # Combine extended + training data for metrics.csv
    print("\nCombining extended and training data...")
    combined_rows = combine_metrics(extended_rows, training_rows)
    print(f"  Total combined rows: {len(combined_rows)}")
    if combined_rows:
        print(f"    Steps: {combined_rows[0]['step']} to {combined_rows[-1]['step']}")
        phases = set(r['phase'] for r in combined_rows)
        for phase in sorted(phases):
            count = sum(1 for r in combined_rows if r['phase'] == phase)
            print(f"    Phase '{phase}': {count} rows")

    # Write output files
    print(f"\nWriting {TOURNAMENT_CSV}...")
    write_tournament_csv(tournament_rows)
    print(f"  Wrote {len(tournament_rows)} rows")

    print(f"\nWriting {METRICS_CSV}...")
    write_metrics_csv(combined_rows)
    print(f"  Wrote {len(combined_rows)} rows")

    print(f"\nWriting {SAMPLES_JSONL}...")
    write_samples_jsonl(samples_list)
    print(f"  Wrote {len(samples_list)} sample events")

    # Verification
    print("\n" + "=" * 60)
    print("  Verification")
    print("=" * 60)

    for path in [TOURNAMENT_CSV, METRICS_CSV, SAMPLES_JSONL]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            with open(path, 'r') as f:
                line_count = sum(1 for _ in f)
            print(f"  {os.path.basename(path):30s} {line_count:6d} lines  {size:,} bytes")
        else:
            print(f"  {os.path.basename(path):30s} MISSING!")

    # Quick sanity: print first and last row of metrics.csv
    print("\n  metrics.csv first data row:")
    with open(METRICS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            r = rows[0]
            print(f"    step={r['step']} loss={r['loss']} phase={r['phase']}")
            r = rows[-1]
            print(f"  metrics.csv last data row:")
            print(f"    step={r['step']} loss={r['loss']} phase={r['phase']}")

    # Quick sanity: print first sample
    if os.path.exists(SAMPLES_JSONL):
        with open(SAMPLES_JSONL, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                s = json.loads(first_line)
                print(f"\n  First sample event: step={s['step']}, "
                      f"{len(s['samples'])} samples, "
                      f"tokens={s.get('tokens_billions', '?')}B")

    print("\nDone!")


if __name__ == "__main__":
    main()
