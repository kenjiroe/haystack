# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Learning Tracker Component for Haystack Learning Project

This component tracks progress through the 12-week learning journey,
providing progress reports and completion metrics following Haystack standards.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from haystack import component, default_to_dict, default_from_dict, logging
from haystack.core.errors import ComponentError

logger = logging.getLogger(__name__)


class LearningTrackerError(ComponentError):
    """Exception raised when LearningTracker encounters an error."""
    pass


@component
class LearningTracker:
    """
    A component to track learning progress through the Haystack learning journey.

    This component maintains a record of completed exercises, tracks completion rates,
    and provides progress reports for each phase of the learning plan.

    Usage example:
    ```python
    from learning_phases.phase_01.learning_tracker import LearningTracker

    tracker = LearningTracker()
    result = tracker.run(
        exercise_name="Basic Pipeline Creation",
        phase="Phase 1",
        total_exercises=12
    )
    print(result["progress_report"])
    ```
    """

    def __init__(
        self,
        progress_file: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize the Learning Tracker.

        :param progress_file: Path to save progress data (optional)
        :param auto_save: Whether to automatically save progress to file
        """
        self.progress_file = Path(progress_file) if progress_file else Path("learning_progress.json")
        self.auto_save = auto_save
        self.completed_exercises: List[Dict[str, Any]] = []
        self.phase_progress: Dict[str, int] = {}

        # Load existing progress if file exists
        self._load_progress()

        logger.info(f"LearningTracker initialized with {len(self.completed_exercises)} completed exercises")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with the component's configuration and state.
        """
        return default_to_dict(
            self,
            progress_file=str(self.progress_file) if self.progress_file else None,
            auto_save=self.auto_save,
            completed_exercises=self.completed_exercises,
            phase_progress=self.phase_progress,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningTracker":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize the component from.
        :returns: Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if "progress_file" in init_params and init_params["progress_file"] is not None:
            init_params["progress_file"] = str(init_params["progress_file"])

        instance = default_from_dict(cls, data)

        # Restore state
        if "completed_exercises" in data:
            instance.completed_exercises = data["completed_exercises"]
        if "phase_progress" in data:
            instance.phase_progress = data["phase_progress"]

        return instance

    def warm_up(self) -> None:
        """
        Warm up the component by validating configuration and loading progress.
        """
        try:
            # Validate progress file path
            if self.progress_file and not self.progress_file.parent.exists():
                self.progress_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing progress
            self._load_progress()

            logger.info("LearningTracker warmed up successfully")

        except Exception as e:
            raise LearningTrackerError(f"Failed to warm up LearningTracker: {e}") from e

    @component.output_types(
        progress_report=str,
        completion_rate=float,
        phase_progress=Dict[str, float],
        total_completed=int,
        recommendations=List[str]
    )
    def run(
        self,
        exercise_name: str,
        phase: str = "Phase 1",
        total_exercises: int = 12,
        difficulty_level: str = "beginner",
        estimated_time: int = 60  # minutes
    ) -> Dict[str, Any]:
        """
        Track completion of an exercise and return progress information.

        :param exercise_name: Name of the completed exercise
        :param phase: Current learning phase (e.g., "Phase 1", "Phase 2")
        :param total_exercises: Total number of exercises in current phase
        :param difficulty_level: Difficulty level (beginner, intermediate, advanced)
        :param estimated_time: Estimated completion time in minutes

        :returns: Dictionary containing progress information

        :raises LearningTrackerError: If tracking fails due to invalid inputs
        """
        # Validate inputs
        if not exercise_name or not isinstance(exercise_name, str):
            raise LearningTrackerError("exercise_name must be a non-empty string")

        if total_exercises <= 0:
            raise LearningTrackerError("total_exercises must be positive")

        if difficulty_level not in ["beginner", "intermediate", "advanced"]:
            raise LearningTrackerError("difficulty_level must be 'beginner', 'intermediate', or 'advanced'")

        try:
            # Record the completed exercise
            exercise_entry = {
                "name": exercise_name,
                "phase": phase,
                "difficulty": difficulty_level,
                "estimated_time": estimated_time,
                "completed_at": datetime.now().isoformat(),
                "session_id": f"{phase}_{len(self.completed_exercises) + 1}"
            }

            self.completed_exercises.append(exercise_entry)

            # Update phase progress
            if phase not in self.phase_progress:
                self.phase_progress[phase] = 0
            self.phase_progress[phase] += 1

            # Calculate metrics
            phase_completion_rate = self.phase_progress[phase] / total_exercises
            total_completed = len(self.completed_exercises)
            overall_completion_rate = total_completed / (total_exercises * 6)  # 6 phases total

            # Generate progress report
            progress_report = self._generate_progress_report(
                exercise_name, phase, phase_completion_rate, total_completed
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                phase, phase_completion_rate, difficulty_level
            )

            # Calculate phase progress percentages
            phase_progress_percentages = {
                p: (count / total_exercises) * 100
                for p, count in self.phase_progress.items()
            }

            # Auto-save progress if enabled
            if self.auto_save:
                self._save_progress()

            logger.info(f"Exercise '{exercise_name}' completed. Phase progress: {phase_completion_rate:.1%}")

            return {
                "progress_report": progress_report,
                "completion_rate": phase_completion_rate,
                "phase_progress": phase_progress_percentages,
                "total_completed": total_completed,
                "recommendations": recommendations
            }

        except Exception as e:
            if isinstance(e, LearningTrackerError):
                raise
            raise LearningTrackerError(f"Failed to track exercise completion: {e}") from e

    def _generate_progress_report(
        self,
        exercise_name: str,
        phase: str,
        completion_rate: float,
        total_completed: int
    ) -> str:
        """Generate a detailed progress report."""
        recent_exercises = [ex["name"] for ex in self.completed_exercises[-3:]]

        report = f"""
🎯 Learning Progress Report
==========================

✅ Just Completed: {exercise_name}
📚 Current Phase: {phase}
📈 Phase Progress: {completion_rate:.1%}
🏆 Total Exercises: {total_completed}

📝 Recent Completions:
{chr(10).join(f"   • {ex}" for ex in recent_exercises)}

⏱️  Session Summary:
   • Phase: {phase}
   • Completion Rate: {completion_rate:.1%}
   • Status: {'✅ Phase Complete!' if completion_rate >= 1.0 else '🚧 In Progress'}
"""
        return report.strip()

    def _generate_recommendations(
        self,
        phase: str,
        completion_rate: float,
        difficulty_level: str
    ) -> List[str]:
        """Generate personalized learning recommendations."""
        recommendations = []

        if completion_rate < 0.3:
            recommendations.append("🌱 Keep building fundamentals - you're doing great!")
            recommendations.append("📖 Review documentation for concepts you find challenging")
        elif completion_rate < 0.7:
            recommendations.append("🚀 Great progress! Consider exploring advanced topics")
            recommendations.append("🤝 Share your work with the community for feedback")
        elif completion_rate < 1.0:
            recommendations.append("🎯 You're almost done with this phase!")
            recommendations.append("🔬 Start thinking about the next phase challenges")
        else:
            recommendations.append("🎉 Phase complete! Ready for the next challenge")
            recommendations.append("📋 Review what you learned before moving forward")

        if difficulty_level == "beginner":
            recommendations.append("💡 Try intermediate exercises when ready")
        elif difficulty_level == "advanced":
            recommendations.append("🏗️ Consider creating your own custom components")

        return recommendations

    def _save_progress(self) -> None:
        """Save progress to file."""
        try:
            progress_data = {
                "completed_exercises": self.completed_exercises,
                "phase_progress": self.phase_progress,
                "last_updated": datetime.now().isoformat(),
                "total_sessions": len(self.completed_exercises)
            }

            with open(self.progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)

            logger.debug(f"Progress saved to {self.progress_file}")

        except Exception as e:
            logger.warning(f"Failed to save progress: {e}")

    def _load_progress(self) -> None:
        """Load progress from file if it exists."""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, "r") as f:
                    data = json.load(f)

                self.completed_exercises = data.get("completed_exercises", [])
                self.phase_progress = data.get("phase_progress", {})

                logger.info(f"Loaded progress from {self.progress_file}")
            else:
                logger.info("No existing progress file found - starting fresh")

        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")

    def get_phase_summary(self, phase: str = None) -> Dict[str, Any]:
        """
        Get summary statistics for a specific phase or all phases.

        :param phase: Phase name to summarize (optional - returns all if None)
        :returns: Summary statistics
        """
        if phase:
            phase_exercises = [ex for ex in self.completed_exercises if ex["phase"] == phase]
            return {
                "phase": phase,
                "completed_count": len(phase_exercises),
                "exercises": [ex["name"] for ex in phase_exercises],
                "total_time": sum(ex["estimated_time"] for ex in phase_exercises),
                "difficulty_breakdown": self._get_difficulty_breakdown(phase_exercises)
            }
        else:
            return {
                "total_completed": len(self.completed_exercises),
                "phases": list(self.phase_progress.keys()),
                "phase_progress": self.phase_progress,
                "total_time": sum(ex["estimated_time"] for ex in self.completed_exercises)
            }

    def _get_difficulty_breakdown(self, exercises: List[Dict]) -> Dict[str, int]:
        """Get breakdown of exercises by difficulty level."""
        breakdown = {"beginner": 0, "intermediate": 0, "advanced": 0}
        for ex in exercises:
            difficulty = ex.get("difficulty", "beginner")
            breakdown[difficulty] = breakdown.get(difficulty, 0) + 1
        return breakdown

    def reset_progress(self) -> None:
        """Reset all progress (useful for testing)."""
        self.completed_exercises = []
        self.phase_progress = {}
        if self.auto_save and self.progress_file.exists():
            self.progress_file.unlink()
        logger.info("Progress reset successfully")


# Helper function for easy component creation
def create_learning_tracker(progress_file: str = None) -> LearningTracker:
    """
    Factory function to create a LearningTracker component.

    :param progress_file: Optional path to progress file
    :returns: Configured LearningTracker instance
    """
    return LearningTracker(progress_file=progress_file)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    tracker = LearningTracker(progress_file="test_progress.json")

    # Warm up component
    tracker.warm_up()

    # Simulate completing some exercises
    result = tracker.run(
        exercise_name="Environment Setup",
        phase="Phase 1",
        total_exercises=12,
        difficulty_level="beginner",
        estimated_time=120
    )

    print(result["progress_report"])
    print(f"Recommendations: {result['recommendations']}")

    # Complete another exercise
    result = tracker.run(
        exercise_name="First Pipeline Creation",
        phase="Phase 1",
        total_exercises=12,
        difficulty_level="beginner",
        estimated_time=90
    )

    print(result["progress_report"])

    # Get phase summary
    summary = tracker.get_phase_summary("Phase 1")
    print(f"Phase 1 Summary: {summary}")

    # Test serialization
    tracker_dict = tracker.to_dict()
    tracker_restored = LearningTracker.from_dict(tracker_dict)
    print(f"Serialization test: {len(tracker_restored.completed_exercises)} exercises restored")
