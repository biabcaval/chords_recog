"""
Curriculum Learning Module for Chord Recognition

This module implements various curriculum learning strategies to train
the model progressively from easier to harder examples.
"""

import numpy as np
import torch
from collections import Counter
import os


class CurriculumLearning:
    """
    Implements curriculum learning strategies for chord recognition.
    
    Difficulty metrics:
    1. Chord Complexity: Simple chords (maj, min) are easier than complex (7th, 9th, sus, etc.)
    2. Change Frequency: Songs with fewer chord changes are easier
    3. Unique Chords: Songs with fewer unique chords are easier
    4. Mixed: Combination of all metrics
    """
    
    # Chord complexity weights (lower = easier)
    CHORD_COMPLEXITY = {
        'maj': 1.0,
        'min': 1.0,
        '7': 2.0,
        'maj7': 2.5,
        'min7': 2.5,
        'maj6': 2.0,
        'min6': 2.0,
        'sus2': 2.5,
        'sus4': 2.5,
        'dim': 3.0,
        'aug': 3.0,
        'hdim7': 3.5,
        'dim7': 3.5,
        '9': 3.0,
        'maj9': 3.5,
        'min9': 3.5,
        'N': 0.5,  # No chord is easiest
        'X': 4.0,  # Unknown chord is hardest
    }
    
    def __init__(self, config, dataset, logger=None):
        """
        Initialize curriculum learning module.
        
        Args:
            config: Configuration object with curriculum settings
            dataset: Training dataset
            logger: Logger for debugging
        """
        self.config = config
        self.dataset = dataset
        self.logger = logger
        self.curriculum_config = config.curriculum
        
        if not self.curriculum_config['enabled']:
            return
        
        self.strategy = self.curriculum_config['strategy']
        self.pacing = self.curriculum_config['pacing']
        self.start_ratio = self.curriculum_config['start_ratio']
        self.pace_epochs = self.curriculum_config['pace_epochs']
        self.warmup_epochs = self.curriculum_config['warmup_epochs']
        
        # Compute difficulty scores for all samples
        self.log("Computing difficulty scores for curriculum learning...")
        self.difficulty_scores = self._compute_difficulty_scores()
        self.sorted_indices = np.argsort(self.difficulty_scores)
        self.log(f"Difficulty scores computed. Range: [{self.difficulty_scores.min():.3f}, {self.difficulty_scores.max():.3f}]")
        
    def log(self, message):
        """Log message if logger is available."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _compute_difficulty_scores(self):
        """
        Compute difficulty score for each sample in the dataset.
        
        Returns:
            np.array: Difficulty scores for each sample (0=easiest, 1=hardest)
        """
        num_samples = len(self.dataset)
        scores = np.zeros(num_samples)
        
        for idx in range(num_samples):
            try:
                sample = self.dataset[idx]
                chords = sample['chord']
                
                if self.strategy == 'chord_complexity':
                    scores[idx] = self._chord_complexity_score(chords)
                elif self.strategy == 'change_frequency':
                    scores[idx] = self._change_frequency_score(chords)
                elif self.strategy == 'unique_chords':
                    scores[idx] = self._unique_chords_score(chords)
                elif self.strategy == 'mixed':
                    scores[idx] = self._mixed_score(chords)
                else:
                    raise ValueError(f"Unknown curriculum strategy: {self.strategy}")
                    
            except Exception as e:
                self.log(f"Error computing difficulty for sample {idx}: {e}")
                scores[idx] = 0.5  # Default to medium difficulty
        
        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores
    
    def _chord_complexity_score(self, chords):
        """
        Compute difficulty based on chord complexity.
        
        Args:
            chords: Array of chord indices
            
        Returns:
            float: Complexity score
        """
        # Convert chord indices to chord types and compute average complexity
        from utils.mir_eval_modules import idx2voca_chord
        
        idx2chord_map = idx2voca_chord()
        complexities = []
        
        for chord_idx in chords:
            chord_label = idx2chord_map.get(chord_idx, 'X')
            
            # Extract chord type (e.g., 'A:maj7' -> 'maj7', 'B' -> 'maj')
            if chord_label == 'N':
                chord_type = 'N'
            elif chord_label == 'X':
                chord_type = 'X'
            elif ':' in chord_label:
                chord_type = chord_label.split(':')[1].split('/')[0]
            else:
                chord_type = 'maj'  # Default for root-only chords
            
            # Get complexity weight
            complexity = self.CHORD_COMPLEXITY.get(chord_type, 2.0)
            complexities.append(complexity)
        
        return np.mean(complexities)
    
    def _change_frequency_score(self, chords):
        """
        Compute difficulty based on chord change frequency.
        Higher change frequency = harder
        
        Args:
            chords: Array of chord indices
            
        Returns:
            float: Change frequency score
        """
        if len(chords) <= 1:
            return 0.0
        
        # Count chord changes
        changes = np.sum(np.diff(chords) != 0)
        change_rate = changes / len(chords)
        
        return change_rate
    
    def _unique_chords_score(self, chords):
        """
        Compute difficulty based on number of unique chords.
        More unique chords = harder
        
        Args:
            chords: Array of chord indices
            
        Returns:
            float: Unique chords score
        """
        num_unique = len(np.unique(chords))
        
        # Normalize by a reasonable maximum (e.g., 25 for majmin, 170 for large voca)
        max_chords = self.config.model['num_chords']
        return min(num_unique / max_chords, 1.0)
    
    def _mixed_score(self, chords):
        """
        Compute difficulty using a weighted combination of all metrics.
        
        Args:
            chords: Array of chord indices
            
        Returns:
            float: Mixed difficulty score
        """
        complexity = self._chord_complexity_score(chords)
        change_freq = self._change_frequency_score(chords)
        unique = self._unique_chords_score(chords)
        
        # Weighted combination
        weights = {
            'complexity': 0.4,
            'change_freq': 0.3,
            'unique': 0.3
        }
        
        score = (weights['complexity'] * complexity + 
                weights['change_freq'] * change_freq + 
                weights['unique'] * unique)
        
        return score
    
    def get_curriculum_ratio(self, epoch):
        """
        Get the ratio of data to include at the given epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            float: Ratio of data to include (0.0 to 1.0)
        """
        if not self.curriculum_config['enabled']:
            return 1.0
        
        # Warmup period: use only easiest samples
        if epoch < self.warmup_epochs:
            return self.start_ratio
        
        # Calculate progress through curriculum
        progress_epoch = epoch - self.warmup_epochs
        if progress_epoch >= self.pace_epochs:
            return 1.0
        
        progress = progress_epoch / self.pace_epochs
        
        # Apply pacing function
        if self.pacing == 'linear':
            paced_progress = progress
        elif self.pacing == 'quadratic':
            paced_progress = progress ** 2
        elif self.pacing == 'exponential':
            paced_progress = (np.exp(progress) - 1) / (np.e - 1)
        elif self.pacing == 'step':
            # Step function with 4 steps
            steps = 4
            paced_progress = np.floor(progress * steps) / steps
        else:
            paced_progress = progress
        
        # Calculate current ratio
        ratio = self.start_ratio + (1.0 - self.start_ratio) * paced_progress
        return ratio
    
    def get_sample_indices(self, epoch):
        """
        Get the indices of samples to use at the given epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            np.array: Indices of samples to include
        """
        if not self.curriculum_config['enabled']:
            return np.arange(len(self.dataset))
        
        ratio = self.get_curriculum_ratio(epoch)
        num_samples = int(len(self.dataset) * ratio)
        
        # Return indices sorted by difficulty (easiest first)
        return self.sorted_indices[:num_samples]
    
    def get_curriculum_sampler(self, epoch):
        """
        Get a sampler for curriculum learning at the given epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            torch.utils.data.SubsetRandomSampler: Sampler for the DataLoader
        """
        if not self.curriculum_config['enabled']:
            return None
        
        indices = self.get_sample_indices(epoch)
        return torch.utils.data.SubsetRandomSampler(indices)
    
    def get_stats(self, epoch):
        """
        Get curriculum learning statistics for the current epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            dict: Statistics dictionary
        """
        if not self.curriculum_config['enabled']:
            return {'enabled': False}
        
        ratio = self.get_curriculum_ratio(epoch)
        indices = self.get_sample_indices(epoch)
        difficulties = self.difficulty_scores[indices]
        
        return {
            'enabled': True,
            'epoch': epoch,
            'ratio': ratio,
            'num_samples': len(indices),
            'total_samples': len(self.dataset),
            'mean_difficulty': difficulties.mean(),
            'max_difficulty': difficulties.max(),
            'min_difficulty': difficulties.min(),
            'strategy': self.strategy,
            'pacing': self.pacing
        }


class CurriculumDataLoader:
    """
    DataLoader wrapper that implements curriculum learning.
    """
    
    def __init__(self, dataset, curriculum, batch_size, collate_fn, drop_last=False):
        """
        Initialize curriculum dataloader.
        
        Args:
            dataset: Training dataset
            curriculum: CurriculumLearning instance
            batch_size: Batch size
            collate_fn: Collate function
            drop_last: Whether to drop last batch
        """
        self.dataset = dataset
        self.curriculum = curriculum
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.current_epoch = 0
        self.current_loader = None
        
    def set_epoch(self, epoch):
        """
        Set the current epoch and update the dataloader.
        
        Args:
            epoch: Current epoch
        """
        self.current_epoch = epoch
        
        if self.curriculum.curriculum_config['enabled']:
            sampler = self.curriculum.get_curriculum_sampler(epoch)
            self.current_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_fn,
                drop_last=self.drop_last
            )
        else:
            self.current_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                drop_last=self.drop_last
            )
    
    def __iter__(self):
        """Iterate over the current dataloader."""
        if self.current_loader is None:
            self.set_epoch(self.current_epoch)
        return iter(self.current_loader)
    
    def __len__(self):
        """Return length of current dataloader."""
        if self.current_loader is None:
            self.set_epoch(self.current_epoch)
        return len(self.current_loader)

