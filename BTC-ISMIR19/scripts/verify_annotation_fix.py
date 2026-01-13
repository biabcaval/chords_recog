#!/usr/bin/env python
"""
Verify that the annotation fixes resolved the preprocessing errors.
"""
import librosa

def check_preprocessing_condition(audio_path, annotation_end_time, skip_interval=5.0, song_hz=22050):
    """
    Check if the preprocessing condition will pass.
    
    The error occurs when:
        audio_length + skip_interval < last_sec_hz
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=song_hz)
    audio_length = y.shape[0]  # in samples
    
    # Convert annotation end time to samples
    last_sec_hz = int(annotation_end_time * song_hz)
    skip_interval_samples = skip_interval * song_hz
    
    # Check condition
    will_fail = audio_length + skip_interval_samples < last_sec_hz
    
    print(f"\nAudio: {audio_path.split('/')[-1]}")
    print(f"  Audio length: {audio_length / song_hz:.3f}s ({audio_length} samples)")
    print(f"  Annotation end: {annotation_end_time:.3f}s ({last_sec_hz} samples)")
    print(f"  Skip interval: {skip_interval}s ({int(skip_interval_samples)} samples)")
    print(f"  Condition: {audio_length} + {int(skip_interval_samples)} < {last_sec_hz}")
    print(f"            {audio_length + int(skip_interval_samples)} < {last_sec_hz}")
    
    if will_fail:
        print(f"  ✗ WILL FAIL: Song too short by {(last_sec_hz - audio_length - skip_interval_samples) / song_hz:.3f}s")
        return False
    else:
        print(f"  ✓ WILL PASS: Preprocessing will succeed!")
        return True

# Test the previously failing songs
print("="*70)
print("VERIFYING ANNOTATION FIXES")
print("="*70)

test_songs = [
    ("0400-Shirley_Brown-Woman_To_Woman", 234.800),
    ("1138-Mary_MacGregor-Torn_Between_Two_Lovers", 222.507),
]

audio_dir = "/home/daniel.melo/datasets/billboard/audio"
all_passed = True

for song_name, annotation_end in test_songs:
    audio_path = f"{audio_dir}/{song_name}.wav"
    passed = check_preprocessing_condition(audio_path, annotation_end)
    if not passed:
        all_passed = False

print("\n" + "="*70)
if all_passed:
    print("✓ ALL SONGS WILL NOW PREPROCESS SUCCESSFULLY!")
else:
    print("✗ Some songs still have issues")
print("="*70)



