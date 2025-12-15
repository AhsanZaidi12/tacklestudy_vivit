#!/usr/bin/env python3
"""
Fix video file extension mismatches in CSV label files
Automatically processes ALL runs and ALL folds by default
Usage: python fix_video_extensions.py --base_dir /path/to/taguchi_runs --apply
"""
import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

def fix_extensions_in_csv(csv_path, video_dir, dry_run=True):
    """
    Fix file extensions in CSV by checking actual files in video directory
    
    Args:
        csv_path: Path to CSV file (e.g., train_labels.csv)
        video_dir: Directory containing actual video files
        dry_run: If True, only show what would change without modifying
    
    Returns:
        Number of files that needed fixing
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    if 'fname' not in df.columns:
        print(f"    Error: 'fname' column not found in {csv_path}")
        return 0
    
    changes = []
    fixed_count = 0
    
    for idx, row in df.iterrows():
        original_fname = row['fname']
        video_path = os.path.join(video_dir, original_fname)
        
        # Check if file exists
        if os.path.exists(video_path):
            continue  # File exists with current extension, no change needed
        
        # File doesn't exist - try different extensions
        base_name = os.path.splitext(original_fname)[0]
        current_ext = os.path.splitext(original_fname)[1]
        
        # Try .avi if current is .mp4, or .mp4 if current is .avi
        if current_ext == '.mp4':
            new_fname = base_name + '.avi'
        elif current_ext == '.avi':
            new_fname = base_name + '.mp4'
        else:
            # Try both
            for ext in ['.avi', '.mp4']:
                test_path = os.path.join(video_dir, base_name + ext)
                if os.path.exists(test_path):
                    new_fname = base_name + ext
                    break
            else:
                continue
        
        # Check if the new filename exists
        new_path = os.path.join(video_dir, new_fname)
        if os.path.exists(new_path):
            changes.append({
                'index': idx,
                'original': original_fname,
                'corrected': new_fname
            })
            fixed_count += 1
            
            if not dry_run:
                df.at[idx, 'fname'] = new_fname
    
    # Print changes (limited to first 5 to avoid clutter)
    if changes:
        for i, change in enumerate(changes[:5]):
            print(f"      {change['original']} -> {change['corrected']}")
        if len(changes) > 5:
            print(f"      ... and {len(changes) - 5} more")
    
    # Save if not dry run
    if not dry_run and fixed_count > 0:
        # Backup original
        backup_path = csv_path + '.backup'
        if not os.path.exists(backup_path):  # Don't overwrite existing backup
            df_original = pd.read_csv(csv_path)
            df_original.to_csv(backup_path, index=False)
        
        # Save corrected version
        df.to_csv(csv_path, index=False)
    
    return fixed_count

def process_all_runs(base_dir, runs=None, folds=None, dry_run=True):
    """
    Process all runs and folds in the base directory
    
    Args:
        base_dir: Base directory containing run_XX/fold_X structure
        runs: List of run names to process (None = all runs)
        folds: List of fold IDs to process (None = all folds 0-4)
        dry_run: If True, only show what would change
    """
    # Auto-detect all runs if not specified
    if runs is None:
        all_dirs = sorted([d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d)) 
                          and d.startswith('run_')])
        runs = all_dirs
        print(f"Auto-detected {len(runs)} runs: {', '.join(runs[:5])}{', ...' if len(runs) > 5 else ''}")
    
    # Use all folds 0-4 if not specified
    if folds is None:
        folds = list(range(5))
        print(f"Processing all {len(folds)} folds: {folds}")
    
    print("=" * 70)
    print("VIDEO EXTENSION FIXER - BATCH MODE")
    print("=" * 70)
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'LIVE MODE (will modify files)'}")
    print(f"Base directory: {base_dir}")
    print(f"Runs to process: {len(runs)}")
    print(f"Folds per run: {len(folds)}")
    print(f"Total combinations: {len(runs) * len(folds) * 2} (train + val)")
    print("=" * 70)
    
    total_fixed = 0
    total_processed = 0
    runs_with_issues = []
    
    for run_idx, run in enumerate(runs, 1):
        run_has_issues = False
        
        for fold_idx, fold in enumerate(folds, 1):
            fold_dir = os.path.join(base_dir, run, f"fold_{fold}")
            
            if not os.path.exists(fold_dir):
                continue
            
            # Progress indicator
            progress = f"[{run_idx}/{len(runs)}] {run}/fold_{fold} [{fold_idx}/{len(folds)}]"
            print(f"\n{progress}")
            print("-" * 70)
            
            fold_fixed = 0
            
            # Process train labels
            train_csv = os.path.join(fold_dir, "train_labels.csv")
            train_videos = os.path.join(fold_dir, "train", "videos")
            
            if os.path.exists(train_csv) and os.path.exists(train_videos):
                print("  Train:")
                fixed = fix_extensions_in_csv(train_csv, train_videos, dry_run)
                if fixed > 0:
                    print(f"    Fixed: {fixed} files")
                    fold_fixed += fixed
                    run_has_issues = True
                else:
                    print(f"    ✓ All correct")
                total_processed += 1
            
            # Process validation labels
            val_csv = os.path.join(fold_dir, "val_labels.csv")
            val_videos = os.path.join(fold_dir, "val", "videos")
            
            if os.path.exists(val_csv) and os.path.exists(val_videos):
                print("  Validation:")
                fixed = fix_extensions_in_csv(val_csv, val_videos, dry_run)
                if fixed > 0:
                    print(f"    Fixed: {fixed} files")
                    fold_fixed += fixed
                    run_has_issues = True
                else:
                    print(f"    ✓ All correct")
                total_processed += 1
            
            total_fixed += fold_fixed
        
        if run_has_issues:
            runs_with_issues.append(run)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {total_processed} CSV files across {len(runs)} runs")
    print(f"Total files needing correction: {total_fixed}")
    
    if runs_with_issues:
        print(f"\nRuns with extension mismatches ({len(runs_with_issues)}):")
        for run in runs_with_issues:
            print(f"  - {run}")
    
    if dry_run and total_fixed > 0:
        print("\n" + "⚠" * 35)
        print("THIS WAS A DRY RUN - NO FILES WERE MODIFIED")
        print("To apply changes, run again with --apply flag:")
        print(f"  python {os.path.basename(__file__)} --base_dir {base_dir} --apply")
        print("⚠" * 35)
    elif not dry_run and total_fixed > 0:
        print("\n" + "✓" * 35)
        print("CHANGES APPLIED SUCCESSFULLY")
        print("Backups created with .backup extension")
        print("✓" * 35)
    else:
        print("\n✓ All files already have correct extensions")
    
    print("=" * 70)
    
    return total_fixed

def main():
    parser = argparse.ArgumentParser(
        description='Fix video file extension mismatches in CSV label files\n'
                    'Processes ALL runs and ALL folds by default',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes for ALL runs and folds (DRY RUN)
  python fix_video_extensions.py --base_dir /path/to/taguchi_runs
  
  # Apply changes to ALL runs and folds
  python fix_video_extensions.py --base_dir /path/to/taguchi_runs --apply
  
  # Process specific runs only
  python fix_video_extensions.py --base_dir /path/to/taguchi_runs --runs run_05,run_06 --apply
  
  # Process specific folds only  
  python fix_video_extensions.py --base_dir /path/to/taguchi_runs --folds 0,1,2 --apply

Note: By default, ALL runs and ALL folds are processed.
      Use --runs or --folds to limit the scope.
        """
    )
    
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing run_XX/fold_X structure')
    parser.add_argument('--runs', type=str, default=None,
                       help='Comma-separated run names (default: ALL runs)')
    parser.add_argument('--folds', type=str, default=None,
                       help='Comma-separated fold IDs (default: ALL folds 0-4)')
    parser.add_argument('--apply', action='store_true',
                       help='Apply changes (default is dry run)')
    
    args = parser.parse_args()
    
    # Validate base directory
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        return
    
    # Parse runs and folds
    runs = None
    if args.runs:
        runs = [r.strip() for r in args.runs.split(',')]
        print(f"Limiting to specified runs: {runs}")
    
    folds = None
    if args.folds:
        folds = [int(f.strip()) for f in args.folds.split(',')]
        print(f"Limiting to specified folds: {folds}")
    
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Process
    total_fixed = process_all_runs(
        args.base_dir,
        runs=runs,
        folds=folds,
        dry_run=not args.apply
    )
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if total_fixed > 0 and not args.apply:
        print("\nNext step: Run with --apply to make changes permanent")

if __name__ == "__main__":
    main()