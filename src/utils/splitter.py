"""
Dataset splitter for train/validation splits.
Supports random and weighted strategies.
"""

import random
from typing import List, Tuple, Dict
from collections import defaultdict


class Splitter:
    """Split datasets into train/validation"""

    def split_dataset(
        self,
        samples: List,
        ratio: float = 0.8,
        strategy: str = "random",
        weight_by: str = "source"
    ) -> Tuple[List, List]:
        """
        Split dataset into train and validation sets.

        Args:
            samples: List of Sample objects
            ratio: Train ratio (0.8 = 80% train, 20% val)
            strategy: "random" or "weighted"
            weight_by: "source" or "keywords" (if weighted)

        Returns:
            (train_samples, val_samples)
        """
        if not samples:
            return [], []

        if strategy == "random":
            return self._random_split(samples, ratio)
        elif strategy == "weighted":
            return self._weighted_split(samples, ratio, weight_by)
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Must be 'random' or 'weighted'"
            )

    def _random_split(
        self,
        samples: List,
        ratio: float
    ) -> Tuple[List, List]:
        """
        Random split with shuffling.

        Args:
            samples: List of samples
            ratio: Train ratio

        Returns:
            (train, val)
        """
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)

        # Split
        split_idx = int(len(shuffled) * ratio)

        train = shuffled[:split_idx]
        val = shuffled[split_idx:]

        return train, val

    def _weighted_split(
        self,
        samples: List,
        ratio: float,
        weight_by: str
    ) -> Tuple[List, List]:
        """
        Weighted split ensuring equal representation.

        Strategy: Split each group proportionally to maintain
        equal representation in both train and val sets.

        Example:
        If weight_by="source" and you have:
        - 50 samples from file_a.pdf
        - 30 samples from file_b.pdf
        - 20 samples from file_c.pdf

        With ratio=0.8:
        - file_a: 40 train, 10 val
        - file_b: 24 train, 6 val
        - file_c: 16 train, 4 val

        Args:
            samples: List of samples
            ratio: Train ratio
            weight_by: "source" or "keywords"

        Returns:
            (train, val)
        """
        # Group samples
        if weight_by == "source":
            groups = self._group_by_source(samples)
        elif weight_by == "keywords":
            groups = self._group_by_keywords(samples)
        else:
            raise ValueError(
                f"Unknown weight_by: {weight_by}. "
                f"Must be 'source' or 'keywords'"
            )

        train_all = []
        val_all = []

        # Split each group proportionally
        for group_name, group_samples in groups.items():
            # Shuffle within group
            shuffled = group_samples.copy()
            random.shuffle(shuffled)

            # Calculate split point
            split_idx = int(len(shuffled) * ratio)

            # Ensure at least 1 sample in each split if possible
            if len(shuffled) > 1 and split_idx == 0:
                split_idx = 1
            elif len(shuffled) > 1 and split_idx == len(shuffled):
                split_idx = len(shuffled) - 1

            # Split
            train = shuffled[:split_idx]
            val = shuffled[split_idx:]

            train_all.extend(train)
            val_all.extend(val)

        # Final shuffle to mix groups
        random.shuffle(train_all)
        random.shuffle(val_all)

        return train_all, val_all

    def _group_by_source(self, samples: List) -> Dict[str, List]:
        """
        Group samples by source file.

        Args:
            samples: List of samples

        Returns:
            Dict mapping source -> samples
        """
        groups = defaultdict(list)

        for sample in samples:
            source = sample.source_file
            groups[source].append(sample)

        return dict(groups)

    def _group_by_keywords(self, samples: List) -> Dict[str, List]:
        """
        Group samples by primary keyword.

        Args:
            samples: List of samples

        Returns:
            Dict mapping keyword -> samples
        """
        groups = defaultdict(list)

        for sample in samples:
            # Use first keyword as primary
            if sample.topic_keywords:
                keyword = sample.topic_keywords[0]
            else:
                keyword = "unknown"

            groups[keyword].append(sample)

        return dict(groups)

    def print_split_summary(
        self,
        train: List,
        val: List,
        weight_by: str = None
    ):
        """
        Print summary of split distribution.

        Args:
            train: Train samples
            val: Validation samples
            weight_by: What was weighted by (for detailed stats)
        """
        total = len(train) + len(val)

        print(f"\n{'='*50}")
        print("Split Summary")
        print(f"{'='*50}")
        print(f"Total samples: {total}")
        print(f"Train: {len(train)} ({len(train)/total*100:.1f}%)")
        print(f"Validation: {len(val)} ({len(val)/total*100:.1f}%)")

        if weight_by == "source":
            print(f"\n{'='*50}")
            print("Distribution by Source File")
            print(f"{'='*50}")

            # Get all sources
            all_samples = train + val
            sources = set(s.source_file for s in all_samples)

            for source in sorted(sources):
                train_count = sum(1 for s in train if s.source_file == source)
                val_count = sum(1 for s in val if s.source_file == source)
                total_count = train_count + val_count

                print(f"\n{source}:")
                print(f"  Total: {total_count}")
                print(f"  Train: {train_count} ({train_count/total_count*100:.1f}%)")
                print(f"  Val:   {val_count} ({val_count/total_count*100:.1f}%)")

        elif weight_by == "keywords":
            print(f"\n{'='*50}")
            print("Distribution by Keywords")
            print(f"{'='*50}")

            # Get all keywords
            all_samples = train + val
            keywords = set()
            for s in all_samples:
                if s.topic_keywords:
                    keywords.add(s.topic_keywords[0])

            for keyword in sorted(keywords):
                train_count = sum(
                    1 for s in train
                    if s.topic_keywords and s.topic_keywords[0] == keyword
                )
                val_count = sum(
                    1 for s in val
                    if s.topic_keywords and s.topic_keywords[0] == keyword
                )
                total_count = train_count + val_count

                if total_count > 0:
                    print(f"\n{keyword}:")
                    print(f"  Total: {total_count}")
                    print(f"  Train: {train_count} ({train_count/total_count*100:.1f}%)")
                    print(f"  Val:   {val_count} ({val_count/total_count*100:.1f}%)")

        print(f"\n{'='*50}\n")

    def get_split_statistics(
        self,
        train: List,
        val: List
    ) -> dict:
        """
        Get detailed statistics about the split.

        Args:
            train: Train samples
            val: Validation samples

        Returns:
            Dict with statistics
        """
        total = len(train) + len(val)

        stats = {
            "total_samples": total,
            "train_samples": len(train),
            "val_samples": len(val),
            "train_ratio": len(train) / total if total > 0 else 0,
            "val_ratio": len(val) / total if total > 0 else 0
        }

        # Source distribution
        train_sources = {}
        val_sources = {}

        for s in train:
            source = s.source_file
            train_sources[source] = train_sources.get(source, 0) + 1

        for s in val:
            source = s.source_file
            val_sources[source] = val_sources.get(source, 0) + 1

        stats["train_source_distribution"] = train_sources
        stats["val_source_distribution"] = val_sources

        # Token statistics
        if train:
            stats["train_total_tokens"] = sum(s.token_length for s in train)
            stats["train_avg_tokens"] = stats["train_total_tokens"] / len(train)

        if val:
            stats["val_total_tokens"] = sum(s.token_length for s in val)
            stats["val_avg_tokens"] = stats["val_total_tokens"] / len(val)

        return stats
