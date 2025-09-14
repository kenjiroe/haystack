# SPDX-FileCopyrightText: 2024-present Learning Project <learning@example.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Learning Tracker Component

This module contains comprehensive tests for the LearningTracker component,
following Haystack's testing guidelines with unit, integration, and performance tests.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from learning_phases.phase_01.learning_tracker import LearningTracker, create_learning_tracker


class TestLearningTracker:
    """Unit tests for LearningTracker component."""

    @pytest.fixture
    def temp_progress_file(self):
        """Create a temporary file for progress storage."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        yield temp_file
        # Cleanup
        temp_path = Path(temp_file)
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def tracker(self, temp_progress_file):
        """Create a LearningTracker instance with temporary file."""
        return LearningTracker(progress_file=temp_progress_file, auto_save=True)

    @pytest.fixture
    def tracker_no_save(self):
        """Create a LearningTracker instance without auto-save."""
        return LearningTracker(auto_save=False)

    def test_component_initialization(self, tracker):
        """Test that the component initializes correctly."""
        assert hasattr(tracker, 'completed_exercises')
        assert hasattr(tracker, 'phase_progress')
        assert tracker.completed_exercises == []
        assert tracker.phase_progress == {}
        assert tracker.auto_save is True
        assert tracker.progress_file is not None

    def test_component_has_correct_outputs(self, tracker):
        """Test that component has the expected output types."""
        # Check that the component is properly decorated
        assert hasattr(tracker, '__haystack_output__')

        # Check output socket types
        outputs = tracker.__haystack_output__._sockets_dict
        expected_outputs = [
            'progress_report', 'completion_rate', 'phase_progress',
            'total_completed', 'recommendations'
        ]

        for output in expected_outputs:
            assert output in outputs

    def test_run_with_single_exercise(self, tracker_no_save):
        """Test running the component with a single exercise."""
        result = tracker_no_save.run(
            exercise_name="Test Exercise",
            phase="Phase 1",
            total_exercises=10,
            difficulty_level="beginner",
            estimated_time=60
        )

        # Check return structure
        assert isinstance(result, dict)
        assert 'progress_report' in result
        assert 'completion_rate' in result
        assert 'phase_progress' in result
        assert 'total_completed' in result
        assert 'recommendations' in result

        # Check values
        assert result['completion_rate'] == 0.1  # 1/10
        assert result['total_completed'] == 1
        assert isinstance(result['progress_report'], str)
        assert isinstance(result['recommendations'], list)
        assert len(result['recommendations']) > 0

    def test_run_with_multiple_exercises(self, tracker_no_save):
        """Test completing multiple exercises."""
        exercises = [
            ("Exercise 1", "beginner", 30),
            ("Exercise 2", "intermediate", 45),
            ("Exercise 3", "advanced", 60)
        ]

        for i, (name, difficulty, time) in enumerate(exercises):
            result = tracker_no_save.run(
                exercise_name=name,
                phase="Phase 1",
                total_exercises=10,
                difficulty_level=difficulty,
                estimated_time=time
            )

            expected_rate = (i + 1) / 10
            assert result['completion_rate'] == expected_rate
            assert result['total_completed'] == i + 1

        # Check that all exercises are recorded
        assert len(tracker_no_save.completed_exercises) == 3
        assert tracker_no_save.phase_progress["Phase 1"] == 3

    def test_different_phases(self, tracker_no_save):
        """Test tracking exercises across different phases."""
        # Phase 1 exercises
        tracker_no_save.run("Phase 1 Exercise 1", "Phase 1", 10, "beginner", 30)
        tracker_no_save.run("Phase 1 Exercise 2", "Phase 1", 10, "beginner", 30)

        # Phase 2 exercises
        tracker_no_save.run("Phase 2 Exercise 1", "Phase 2", 8, "intermediate", 45)

        assert tracker_no_save.phase_progress["Phase 1"] == 2
        assert tracker_no_save.phase_progress["Phase 2"] == 1
        assert len(tracker_no_save.completed_exercises) == 3

    def test_progress_report_content(self, tracker_no_save):
        """Test that progress report contains expected information."""
        result = tracker_no_save.run(
            exercise_name="Test Report Exercise",
            phase="Phase 1",
            total_exercises=5,
            difficulty_level="intermediate",
            estimated_time=90
        )

        report = result['progress_report']
        assert "Test Report Exercise" in report
        assert "Phase 1" in report
        assert "20.0%" in report  # 1/5 = 20%
        assert "✅" in report or "🎯" in report  # Some emoji indicator

    def test_recommendations_generation(self, tracker_no_save):
        """Test that recommendations are generated appropriately."""
        # Test beginner recommendations
        result = tracker_no_save.run("Exercise", "Phase 1", 10, "beginner", 30)
        recommendations = result['recommendations']

        assert len(recommendations) > 0
        assert any("intermediate" in rec.lower() for rec in recommendations)

        # Test advanced recommendations
        result = tracker_no_save.run("Advanced Exercise", "Phase 1", 10, "advanced", 60)
        recommendations = result['recommendations']

        assert any("custom" in rec.lower() or "component" in rec.lower() for rec in recommendations)

    def test_completion_rate_calculation(self, tracker_no_save):
        """Test completion rate calculations."""
        total_exercises = 8

        for i in range(1, 6):  # Complete 5 exercises
            result = tracker_no_save.run(
                f"Exercise {i}", "Phase 1", total_exercises, "beginner", 30
            )
            expected_rate = i / total_exercises
            assert abs(result['completion_rate'] - expected_rate) < 0.001

    def test_phase_progress_percentages(self, tracker_no_save):
        """Test phase progress percentage calculations."""
        total_exercises = 10

        # Complete 3 exercises
        for i in range(3):
            result = tracker_no_save.run(
                f"Exercise {i+1}", "Phase 1", total_exercises, "beginner", 30
            )

        phase_progress = result['phase_progress']
        assert "Phase 1" in phase_progress
        assert phase_progress["Phase 1"] == 30.0  # 3/10 * 100

    def test_get_phase_summary(self, tracker_no_save):
        """Test phase summary functionality."""
        # Add some exercises
        tracker_no_save.run("Exercise 1", "Phase 1", 10, "beginner", 30)
        tracker_no_save.run("Exercise 2", "Phase 1", 10, "intermediate", 45)
        tracker_no_save.run("Exercise 3", "Phase 2", 8, "advanced", 60)

        # Test specific phase summary
        phase1_summary = tracker_no_save.get_phase_summary("Phase 1")
        assert phase1_summary['phase'] == "Phase 1"
        assert phase1_summary['completed_count'] == 2
        assert len(phase1_summary['exercises']) == 2
        assert phase1_summary['total_time'] == 75  # 30 + 45

        # Test overall summary
        overall_summary = tracker_no_save.get_phase_summary()
        assert overall_summary['total_completed'] == 3
        assert set(overall_summary['phases']) == {"Phase 1", "Phase 2"}

    def test_difficulty_breakdown(self, tracker_no_save):
        """Test difficulty breakdown calculation."""
        tracker_no_save.run("Easy Exercise", "Phase 1", 10, "beginner", 30)
        tracker_no_save.run("Medium Exercise", "Phase 1", 10, "intermediate", 45)
        tracker_no_save.run("Hard Exercise", "Phase 1", 10, "advanced", 60)

        summary = tracker_no_save.get_phase_summary("Phase 1")
        breakdown = summary['difficulty_breakdown']

        assert breakdown['beginner'] == 1
        assert breakdown['intermediate'] == 1
        assert breakdown['advanced'] == 1

    def test_reset_progress(self, tracker_no_save):
        """Test progress reset functionality."""
        # Add some progress
        tracker_no_save.run("Exercise 1", "Phase 1", 10, "beginner", 30)
        tracker_no_save.run("Exercise 2", "Phase 1", 10, "intermediate", 45)

        assert len(tracker_no_save.completed_exercises) == 2
        assert tracker_no_save.phase_progress["Phase 1"] == 2

        # Reset
        tracker_no_save.reset_progress()

        assert len(tracker_no_save.completed_exercises) == 0
        assert tracker_no_save.phase_progress == {}

    def test_factory_function(self):
        """Test the create_learning_tracker factory function."""
        tracker = create_learning_tracker()
        assert isinstance(tracker, LearningTracker)

        tracker_with_file = create_learning_tracker("test.json")
        assert tracker_with_file.progress_file.name == "test.json"


class TestLearningTrackerFileOperations:
    """Tests for file save/load operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_save_and_load_progress(self, temp_dir):
        """Test saving and loading progress from file."""
        progress_file = temp_dir / "test_progress.json"

        # Create tracker and add some progress
        tracker1 = LearningTracker(progress_file=str(progress_file), auto_save=True)
        tracker1.run("Exercise 1", "Phase 1", 10, "beginner", 30)
        tracker1.run("Exercise 2", "Phase 1", 10, "intermediate", 45)

        # Create new tracker with same file - should load existing progress
        tracker2 = LearningTracker(progress_file=str(progress_file), auto_save=False)

        assert len(tracker2.completed_exercises) == 2
        assert tracker2.phase_progress["Phase 1"] == 2
        assert tracker2.completed_exercises[0]["name"] == "Exercise 1"
        assert tracker2.completed_exercises[1]["name"] == "Exercise 2"

    def test_save_progress_manually(self, temp_dir):
        """Test manual progress saving."""
        progress_file = temp_dir / "manual_progress.json"

        tracker = LearningTracker(progress_file=str(progress_file), auto_save=False)
        tracker.run("Test Exercise", "Phase 1", 10, "beginner", 30)

        # File shouldn't exist yet
        assert not progress_file.exists()

        # Save manually
        tracker._save_progress()

        # File should exist now
        assert progress_file.exists()

        # Verify content
        with open(progress_file) as f:
            data = json.load(f)

        assert len(data["completed_exercises"]) == 1
        assert data["completed_exercises"][0]["name"] == "Test Exercise"

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading when progress file doesn't exist."""
        nonexistent_file = temp_dir / "nonexistent.json"

        # Should not raise an error
        tracker = LearningTracker(progress_file=str(nonexistent_file))

        assert len(tracker.completed_exercises) == 0
        assert tracker.phase_progress == {}

    @patch('learning_phases.phase_01.learning_tracker.logger')
    def test_save_error_handling(self, mock_logger, temp_dir):
        """Test error handling when save fails."""
        # Create a read-only directory to force save failure
        readonly_file = temp_dir / "readonly" / "progress.json"
        readonly_file.parent.mkdir()
        readonly_file.parent.chmod(0o444)  # Read-only

        try:
            tracker = LearningTracker(progress_file=str(readonly_file), auto_save=True)
            tracker.run("Test Exercise", "Phase 1", 10, "beginner", 30)

            # Should have logged a warning
            mock_logger.warning.assert_called()
        finally:
            # Cleanup - restore permissions
            readonly_file.parent.chmod(0o755)

    @patch('learning_phases.phase_01.learning_tracker.logger')
    def test_load_invalid_json(self, mock_logger, temp_dir):
        """Test loading invalid JSON file."""
        progress_file = temp_dir / "invalid.json"

        # Create invalid JSON file
        with open(progress_file, 'w') as f:
            f.write("{ invalid json content")

        # Should not crash, should log warning
        tracker = LearningTracker(progress_file=str(progress_file))

        assert len(tracker.completed_exercises) == 0
        mock_logger.warning.assert_called()


@pytest.mark.integration
class TestLearningTrackerIntegration:
    """Integration tests for LearningTracker component."""

    def test_full_learning_workflow(self):
        """Test a complete learning workflow simulation."""
        tracker = LearningTracker(auto_save=False)

        # Simulate Phase 1 completion
        phase1_exercises = [
            ("Environment Setup", "beginner", 120),
            ("Basic Components", "beginner", 90),
            ("First Pipeline", "intermediate", 60),
            ("Error Handling", "intermediate", 45)
        ]

        for exercise, difficulty, time in phase1_exercises:
            result = tracker.run(
                exercise_name=exercise,
                phase="Phase 1",
                total_exercises=4,
                difficulty_level=difficulty,
                estimated_time=time
            )

        # Check final state
        assert result['completion_rate'] == 1.0  # 100% complete
        assert result['total_completed'] == 4
        assert "Phase Complete" in result['progress_report']

        # Check phase summary
        summary = tracker.get_phase_summary("Phase 1")
        assert summary['completed_count'] == 4
        assert summary['total_time'] == 315  # 120+90+60+45

    def test_multi_phase_workflow(self):
        """Test workflow across multiple phases."""
        tracker = LearningTracker(auto_save=False)

        # Phase 1
        tracker.run("Phase 1 Complete", "Phase 1", 1, "beginner", 60)

        # Phase 2
        tracker.run("Document Processing", "Phase 2", 3, "intermediate", 90)
        tracker.run("Advanced Chunking", "Phase 2", 3, "intermediate", 75)

        # Phase 3
        tracker.run("Search Systems", "Phase 3", 2, "advanced", 120)

        # Verify multi-phase state
        overall = tracker.get_phase_summary()
        assert overall['total_completed'] == 4
        assert len(overall['phases']) == 3
        assert overall['phase_progress']['Phase 1'] == 1
        assert overall['phase_progress']['Phase 2'] == 2
        assert overall['phase_progress']['Phase 3'] == 1


@pytest.mark.integration
def test_real_file_persistence():
    """Test real file persistence with actual files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_file = Path(temp_dir) / "real_progress.json"

        # Session 1 - Create progress
        tracker1 = LearningTracker(progress_file=str(progress_file))
        tracker1.run("Session 1 Exercise", "Phase 1", 10, "beginner", 30)

        # Session 2 - Continue progress
        tracker2 = LearningTracker(progress_file=str(progress_file))
        result = tracker2.run("Session 2 Exercise", "Phase 1", 10, "intermediate", 45)

        # Should have both exercises
        assert result['total_completed'] == 2
        assert tracker2.phase_progress["Phase 1"] == 2

        # Verify file content
        with open(progress_file) as f:
            data = json.load(f)

        assert len(data["completed_exercises"]) == 2
        assert data["total_sessions"] == 2


if __name__ == "__main__":
    # Run basic tests if executed directly
    pytest.main([__file__, "-v"])
