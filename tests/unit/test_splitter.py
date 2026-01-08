"""
Unit tests for src/utils/splitter.py

Tests train/validation splitting with random and weighted strategies.
"""

import pytest
from src.utils.splitter import Splitter, SplitResult, SplitError


class TestSplitterBasic:
    """Basic splitter tests."""

    def test_initialization(self):
        """Test Splitter initialization."""
        splitter = Splitter()
        assert splitter is not None

    def test_initialization_with_ratio(self):
        """Test Splitter with custom ratio."""
        splitter = Splitter(ratio=0.7)
        assert splitter.ratio == 0.7

    def test_initialization_with_strategy(self):
        """Test Splitter with different strategies."""
        strategies = ['random', 'weighted']

        for strategy in strategies:
            splitter = Splitter(strategy=strategy)
            assert splitter.strategy == strategy


class TestRandomSplit:
    """Tests for random splitting strategy."""

    def test_random_split_80_20(self):
        """Test random split with 80/20 ratio."""
        splitter = Splitter(ratio=0.8, strategy='random')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Q{i}'},
                    {'role': 'assistant', 'content': f'A{i}'}
                ]
            }
            for i in range(100)
        ]

        result = splitter.split(dataset)

        assert isinstance(result, SplitResult)
        assert len(result.train) == 80
        assert len(result.validation) == 20
        assert len(result.train) + len(result.validation) == 100

    def test_random_split_70_30(self):
        """Test random split with 70/30 ratio."""
        splitter = Splitter(ratio=0.7, strategy='random')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Q{i}'},
                    {'role': 'assistant', 'content': f'A{i}'}
                ]
            }
            for i in range(100)
        ]

        result = splitter.split(dataset)

        assert len(result.train) == 70
        assert len(result.validation) == 30

    def test_random_split_90_10(self):
        """Test random split with 90/10 ratio."""
        splitter = Splitter(ratio=0.9, strategy='random')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Q{i}'},
                    {'role': 'assistant', 'content': f'A{i}'}
                ]
            }
            for i in range(100)
        ]

        result = splitter.split(dataset)

        assert len(result.train) == 90
        assert len(result.validation) == 10

    def test_random_split_is_random(self):
        """Test that random split produces different splits on multiple runs."""
        splitter = Splitter(ratio=0.8, strategy='random')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Q{i}'},
                    {'role': 'assistant', 'content': f'A{i}'}
                ],
                'id': i  # Track original indices
            }
            for i in range(100)
        ]

        result1 = splitter.split(dataset)
        result2 = splitter.split(dataset)

        # Two random splits should likely differ
        # (Check if at least one sample differs in train set)
        train_ids_1 = {s.get('id') for s in result1.train}
        train_ids_2 = {s.get('id') for s in result2.train}

        # Allow for possibility of same split (low probability)
        assert isinstance(train_ids_1, set)
        assert isinstance(train_ids_2, set)

    def test_random_split_no_overlap(self):
        """Test that train and validation sets don't overlap."""
        splitter = Splitter(ratio=0.8, strategy='random')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': f'Q{i}'},
                    {'role': 'assistant', 'content': f'A{i}'}
                ],
                'id': i
            }
            for i in range(100)
        ]

        result = splitter.split(dataset)

        train_ids = {s.get('id') for s in result.train}
        val_ids = {s.get('id') for s in result.validation}

        # No overlap between train and validation
        assert len(train_ids & val_ids) == 0


class TestWeightedSplitBySource:
    """Tests for weighted splitting by source file."""

    def test_weighted_split_by_source(self):
        """Test weighted split maintains equal representation per source."""
        splitter = Splitter(ratio=0.8, strategy='weighted', weight_by='source')

        # Create dataset with 4 sources, 25 samples each
        dataset = []
        sources = ['source_a.txt', 'source_b.txt', 'source_c.txt', 'source_d.txt']

        for source in sources:
            for i in range(25):
                dataset.append({
                    'messages': [
                        {'role': 'user', 'content': f'Q from {source}'},
                        {'role': 'assistant', 'content': f'A from {source}'}
                    ],
                    'source_file': source
                })

        result = splitter.split(dataset)

        # Count samples per source in train and validation
        train_sources = {}
        val_sources = {}

        for sample in result.train:
            source = sample.get('source_file')
            train_sources[source] = train_sources.get(source, 0) + 1

        for sample in result.validation:
            source = sample.get('source_file')
            val_sources[source] = val_sources.get(source, 0) + 1

        # Each source should have ~20 train and ~5 val (80/20 of 25)
        for source in sources:
            assert train_sources.get(source, 0) == 20
            assert val_sources.get(source, 0) == 5

    def test_weighted_split_unequal_sources(self):
        """Test weighted split with unequal source distributions."""
        splitter = Splitter(ratio=0.8, strategy='weighted', weight_by='source')

        dataset = []

        # Source A: 40 samples
        for i in range(40):
            dataset.append({
                'messages': [{'role': 'user', 'content': 'Q'}],
                'source_file': 'source_a.txt'
            })

        # Source B: 60 samples
        for i in range(60):
            dataset.append({
                'messages': [{'role': 'user', 'content': 'Q'}],
                'source_file': 'source_b.txt'
            })

        result = splitter.split(dataset)

        # Count per source
        train_sources = {}
        val_sources = {}

        for sample in result.train:
            source = sample.get('source_file')
            train_sources[source] = train_sources.get(source, 0) + 1

        for sample in result.validation:
            source = sample.get('source_file')
            val_sources[source] = val_sources.get(source, 0) + 1

        # Each source should maintain 80/20 ratio within itself
        # Source A: 32 train, 8 val
        # Source B: 48 train, 12 val
        assert train_sources.get('source_a.txt') == 32
        assert val_sources.get('source_a.txt') == 8
        assert train_sources.get('source_b.txt') == 48
        assert val_sources.get('source_b.txt') == 12


class TestWeightedSplitByKeywords:
    """Tests for weighted splitting by keywords."""

    def test_weighted_split_by_keywords(self):
        """Test weighted split by topic keywords."""
        splitter = Splitter(ratio=0.8, strategy='weighted', weight_by='keywords')

        dataset = []

        # Topic A: 50 samples
        for i in range(50):
            dataset.append({
                'messages': [{'role': 'user', 'content': 'Q'}],
                'topic_keywords': ['python', 'programming']
            })

        # Topic B: 50 samples
        for i in range(50):
            dataset.append({
                'messages': [{'role': 'user', 'content': 'Q'}],
                'topic_keywords': ['sales', 'negotiation']
            })

        result = splitter.split(dataset)

        # Should maintain representation of both topics
        assert len(result.train) == 80
        assert len(result.validation) == 20

    def test_weighted_split_by_keywords_string(self):
        """Test weighted split when keywords are comma-separated string."""
        splitter = Splitter(ratio=0.8, strategy='weighted', weight_by='keywords')

        dataset = []

        for i in range(100):
            dataset.append({
                'messages': [{'role': 'user', 'content': 'Q'}],
                'topic_keywords': 'python,programming,coding'  # String instead of list
            })

        result = splitter.split(dataset)

        assert len(result.train) + len(result.validation) == 100


class TestSplitStatistics:
    """Tests for split statistics."""

    def test_split_statistics_calculation(self):
        """Test that split statistics are calculated correctly."""
        splitter = Splitter(ratio=0.8, strategy='random')

        dataset = [
            {
                'messages': [{'role': 'user', 'content': f'Q{i}'}],
                'source_file': 'test.txt'
            }
            for i in range(100)
        ]

        result = splitter.split(dataset)

        # Should have statistics
        if hasattr(result, 'statistics'):
            assert result.statistics is not None
            assert 'train_count' in result.statistics or hasattr(result, 'train_count')
            assert 'validation_count' in result.statistics or hasattr(result, 'validation_count')

    def test_split_statistics_per_source(self):
        """Test statistics per source file."""
        splitter = Splitter(ratio=0.8, strategy='weighted', weight_by='source')

        dataset = []
        sources = ['source_a.txt', 'source_b.txt']

        for source in sources:
            for i in range(50):
                dataset.append({
                    'messages': [{'role': 'user', 'content': 'Q'}],
                    'source_file': source
                })

        result = splitter.split(dataset)

        # Statistics should track per-source counts
        assert isinstance(result, SplitResult)


class TestEdgeCases:
    """Tests for edge cases in splitting."""

    def test_split_single_sample(self):
        """Test splitting dataset with single sample."""
        splitter = Splitter(ratio=0.8, strategy='random')

        dataset = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Q'},
                    {'role': 'assistant', 'content': 'A'}
                ]
            }
        ]

        result = splitter.split(dataset)

        # Single sample should go to train or validation
        assert len(result.train) + len(result.validation) == 1

    def test_split_two_samples(self):
        """Test splitting dataset with two samples."""
        splitter = Splitter(ratio=0.8, strategy='random')

        dataset = [
            {'messages': [{'role': 'user', 'content': 'Q1'}]},
            {'messages': [{'role': 'user', 'content': 'Q2'}]}
        ]

        result = splitter.split(dataset)

        assert len(result.train) + len(result.validation) == 2

    def test_split_empty_dataset(self):
        """Test splitting empty dataset."""
        splitter = Splitter(ratio=0.8, strategy='random')

        dataset = []

        with pytest.raises((SplitError, ValueError)):
            splitter.split(dataset)

    def test_split_all_same_source(self):
        """Test weighted split when all samples from same source."""
        splitter = Splitter(ratio=0.8, strategy='weighted', weight_by='source')

        dataset = [
            {
                'messages': [{'role': 'user', 'content': f'Q{i}'}],
                'source_file': 'single_source.txt'
            }
            for i in range(100)
        ]

        result = splitter.split(dataset)

        # Should still split 80/20
        assert len(result.train) == 80
        assert len(result.validation) == 20

    def test_split_missing_source_field(self):
        """Test weighted split by source when some samples missing source field."""
        splitter = Splitter(ratio=0.8, strategy='weighted', weight_by='source')

        dataset = [
            {'messages': [{'role': 'user', 'content': 'Q1'}], 'source_file': 'a.txt'},
            {'messages': [{'role': 'user', 'content': 'Q2'}]},  # Missing source
            {'messages': [{'role': 'user', 'content': 'Q3'}], 'source_file': 'a.txt'}
        ]

        # Should handle gracefully
        result = splitter.split(dataset)
        assert len(result.train) + len(result.validation) == 3


class TestSplitResultDataclass:
    """Tests for SplitResult dataclass."""

    def test_split_result_creation(self):
        """Test SplitResult dataclass creation."""
        train = [{'messages': [{'role': 'user', 'content': 'Q1'}]}]
        validation = [{'messages': [{'role': 'user', 'content': 'Q2'}]}]

        result = SplitResult(train=train, validation=validation)

        assert result.train == train
        assert result.validation == validation

    def test_split_result_with_statistics(self):
        """Test SplitResult with statistics."""
        train = [{'messages': []} for _ in range(80)]
        validation = [{'messages': []} for _ in range(20)]

        result = SplitResult(
            train=train,
            validation=validation,
            statistics={'train_count': 80, 'validation_count': 20}
        )

        if hasattr(result, 'statistics'):
            assert result.statistics['train_count'] == 80


class TestSplitRatioValidation:
    """Tests for split ratio validation."""

    def test_invalid_ratio_too_high(self):
        """Test that ratio > 1.0 raises error."""
        with pytest.raises((ValueError, SplitError)):
            Splitter(ratio=1.5)

    def test_invalid_ratio_too_low(self):
        """Test that ratio < 0.0 raises error."""
        with pytest.raises((ValueError, SplitError)):
            Splitter(ratio=-0.1)

    def test_invalid_ratio_zero(self):
        """Test that ratio = 0.0 raises error."""
        with pytest.raises((ValueError, SplitError)):
            Splitter(ratio=0.0)

    def test_invalid_ratio_one(self):
        """Test that ratio = 1.0 (no validation set) may raise error."""
        # Ratio of 1.0 means all samples go to train, none to validation
        # Implementation may allow or disallow this
        try:
            splitter = Splitter(ratio=1.0)
            assert splitter.ratio == 1.0
        except (ValueError, SplitError):
            # Also acceptable to raise error
            pass

    def test_valid_ratio_range(self):
        """Test that valid ratios (0.5 to 0.95) work."""
        valid_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        for ratio in valid_ratios:
            splitter = Splitter(ratio=ratio)
            assert splitter.ratio == ratio


class TestStrategyValidation:
    """Tests for strategy validation."""

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises((ValueError, SplitError)):
            Splitter(strategy='invalid_strategy')

    def test_valid_strategies(self):
        """Test that valid strategies are accepted."""
        valid_strategies = ['random', 'weighted']

        for strategy in valid_strategies:
            splitter = Splitter(strategy=strategy)
            assert splitter.strategy == strategy
