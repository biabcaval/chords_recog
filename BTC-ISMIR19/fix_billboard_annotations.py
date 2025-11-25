#!/usr/bin/env python
"""
Fix Billboard annotation timestamps to match actual audio durations.

This script:
1. Reads each annotation file in the Billboard dataset
2. Finds the corresponding audio file
3. Gets the actual audio duration
4. Updates the last timestamp in the annotation to match audio duration
5. Creates backups of original files before modifying
"""
import os
import sys
import librosa
import shutil
from pathlib import Path

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        # Just get duration without loading full audio (faster)
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"  ⚠️  Error reading audio: {e}")
        return None

def fix_annotation_file(annotation_path, audio_duration, backup_dir):
    """
    Fix the last timestamp in an annotation file.
    
    Args:
        annotation_path: Path to .lab annotation file
        audio_duration: Actual audio duration in seconds
        backup_dir: Directory to store backups
    """
    # Read annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print(f"  ⚠️  Annotation file too short, skipping")
        return False
    
    # Parse last line with content
    last_line_idx = len(lines) - 1
    while last_line_idx >= 0 and not lines[last_line_idx].strip():
        last_line_idx -= 1
    
    if last_line_idx < 0:
        print(f"  ⚠️  No valid content in annotation file")
        return False
    
    last_line = lines[last_line_idx].strip()
    parts = last_line.split()
    
    if len(parts) < 3:
        print(f"  ⚠️  Invalid annotation format: {last_line}")
        return False
    
    start_time = float(parts[0])
    end_time = float(parts[1])
    chord_label = ' '.join(parts[2:])
    
    print(f"  Original last timestamp: {start_time:.3f} - {end_time:.3f} {chord_label}")
    print(f"  Audio duration: {audio_duration:.3f}s")
    
    # Check if fix is needed
    if end_time <= audio_duration:
        print(f"  ✓ No fix needed (annotation within audio duration)")
        return False
    
    # Create backup
    backup_path = os.path.join(backup_dir, os.path.basename(annotation_path))
    if not os.path.exists(backup_path):
        shutil.copy2(annotation_path, backup_path)
        print(f"  📁 Backup created: {os.path.basename(backup_path)}")
    
    # Update last line with actual audio duration
    new_last_line = f"{start_time:.3f} {audio_duration:.3f} {chord_label}\n"
    lines[last_line_idx] = new_last_line
    
    # Write updated file
    with open(annotation_path, 'w') as f:
        f.writelines(lines)
    
    print(f"  ✓ Fixed: Changed end time from {end_time:.3f}s to {audio_duration:.3f}s")
    return True

def main():
    annotations_dir = "/home/daniel.melo/datasets/billboard/annotations"
    audio_dir = "/home/daniel.melo/datasets/billboard/audio"
    backup_dir = "/home/daniel.melo/datasets/billboard/annotations_backup"
    
    print("="*70)
    print("FIXING BILLBOARD ANNOTATION TIMESTAMPS")
    print("="*70)
    print(f"\nAnnotations: {annotations_dir}")
    print(f"Audio files: {audio_dir}")
    print(f"Backups:     {backup_dir}")
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    print(f"\n✓ Backup directory ready")
    
    # Get all annotation files
    annotation_files = sorted([f for f in os.listdir(annotations_dir) 
                              if f.endswith('.lab')])
    
    print(f"\n📋 Found {len(annotation_files)} annotation files")
    
    # Statistics
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each annotation file
    for i, annotation_file in enumerate(annotation_files, 1):
        print(f"\n[{i}/{len(annotation_files)}] {annotation_file}")
        
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        # Find corresponding audio file
        base_name = os.path.splitext(annotation_file)[0]
        
        # Try different audio extensions
        audio_file = None
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            potential_audio = os.path.join(audio_dir, base_name + ext)
            if os.path.exists(potential_audio):
                audio_file = potential_audio
                break
        
        if not audio_file:
            print(f"  ⚠️  No audio file found, skipping")
            error_count += 1
            continue
        
        # Get audio duration
        audio_duration = get_audio_duration(audio_file)
        if audio_duration is None:
            error_count += 1
            continue
        
        # Fix annotation
        was_fixed = fix_annotation_file(annotation_path, audio_duration, backup_dir)
        if was_fixed:
            fixed_count += 1
        else:
            skipped_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(annotation_files)}")
    print(f"  ✓ Fixed:   {fixed_count}")
    print(f"  - Skipped: {skipped_count} (no fix needed)")
    print(f"  ⚠️  Errors:  {error_count}")
    
    if fixed_count > 0:
        print(f"\n✓ Successfully fixed {fixed_count} annotation files!")
        print(f"  Original files backed up to: {backup_dir}")
    
    if error_count > 0:
        print(f"\n⚠️  {error_count} files had errors (check log above)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()



