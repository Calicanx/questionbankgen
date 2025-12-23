"""Repository for question CRUD operations."""

import logging
from datetime import datetime
from typing import Any, Optional

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from questionbank.mongodb.connection import mongo_db

logger = logging.getLogger(__name__)

# Widget types that should NOT be used for generation
# These require external resources, interactive simulations, or can't be meaningfully varied
EXCLUDED_WIDGET_TYPES = [
    "iframe",           # External simulations - URLs often broken
    "video",            # Video content - can't generate new videos
    "passage",          # Long passages - too complex to vary meaningfully
    "interactive-graph", # Complex interactive widgets
    "grapher",          # Graph plotting tools
    "plotter",          # Plot widgets
    "number-line",      # Interactive number lines
    "orderer",          # Drag-and-drop ordering - renderer issues
    "sorter",           # Drag-and-drop sorting - renderer issues
    "categorizer",      # Drag-and-drop categorization - renderer issues
    "label-image",      # Image labeling - complex interaction
    # "image" - Re-enabled with post-processing URL copy fix
]


class QuestionRepository:
    """Repository for question operations."""

    def __init__(self) -> None:
        self.source = mongo_db.scraped_questions
        self.generated = mongo_db.generated_questions

    def get_question_by_id(self, question_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single question by its MongoDB ObjectId."""
        try:
            if not ObjectId.is_valid(question_id):
                logger.warning(f"Invalid ObjectId format: {question_id}")
                return None

            doc = self.source.find_one({"_id": ObjectId(question_id)})

            if not doc:
                logger.warning(f"Question not found: {question_id}")
                return None

            return doc

        except Exception as e:
            logger.error(f"Error fetching question {question_id}: {e}")
            return None

    def get_random_question(
        self,
        widget_type: Optional[str] = None,
        skill_prefix: Optional[str] = None,
        exclude_widget_types: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Fetch a random question from the source collection.

        Args:
            widget_type: Filter by specific widget type
            skill_prefix: Filter by skill prefix
            exclude_widget_types: If True, exclude problematic widget types (iframe, video, etc.)
        """
        try:
            pipeline = []

            # Build match stage
            match_stage: dict[str, Any] = {}

            if skill_prefix:
                match_stage["skill_prefix"] = {"$regex": f"^{skill_prefix}", "$options": "i"}

            if match_stage:
                pipeline.append({"$match": match_stage})

            # Exclude problematic widget types by default
            if exclude_widget_types and not widget_type:
                # Add fields to check widget types, then exclude problematic ones
                pipeline.extend([
                    {"$addFields": {"widgetsArray": {"$objectToArray": "$question.widgets"}}},
                    {"$match": {"widgetsArray.v.type": {"$nin": EXCLUDED_WIDGET_TYPES}}},
                    {"$project": {"widgetsArray": 0}},
                ])

            if widget_type:
                # Expand widget type aliases
                expanded_types = {widget_type}
                if widget_type == "numeric-input":
                    expanded_types.add("input-number")
                elif widget_type == "input-number":
                    expanded_types.add("numeric-input")

                # Filter by widget type
                pipeline.extend([
                    {"$addFields": {"widgetsArray": {"$objectToArray": "$question.widgets"}}},
                    {"$match": {"widgetsArray.v.type": {"$in": list(expanded_types)}}},
                    {"$project": {"widgetsArray": 0}},
                ])

            # Get random sample
            pipeline.append({"$sample": {"size": 1}})

            cursor = self.source.aggregate(pipeline)
            docs = list(cursor)

            if docs:
                return docs[0]

            return None

        except Exception as e:
            logger.error(f"Error fetching random question: {e}")
            return None

    def get_questions(
        self,
        widget_type: Optional[str] = None,
        skill_prefix: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch multiple questions with optional filters."""
        try:
            pipeline = []

            # Build match stage
            match_stage: dict[str, Any] = {}

            if skill_prefix:
                match_stage["skill_prefix"] = {"$regex": f"^{skill_prefix}", "$options": "i"}

            if match_stage:
                pipeline.append({"$match": match_stage})

            if widget_type:
                expanded_types = {widget_type}
                if widget_type == "numeric-input":
                    expanded_types.add("input-number")
                elif widget_type == "input-number":
                    expanded_types.add("numeric-input")

                pipeline.extend([
                    {"$addFields": {"widgetsArray": {"$objectToArray": "$question.widgets"}}},
                    {"$match": {"widgetsArray.v.type": {"$in": list(expanded_types)}}},
                    {"$project": {"widgetsArray": 0}},
                ])

            # Random sample
            pipeline.append({"$sample": {"size": limit * 2}})
            pipeline.append({"$limit": limit})

            cursor = self.source.aggregate(pipeline)
            return list(cursor)

        except Exception as e:
            logger.error(f"Error fetching questions: {e}")
            return []

    def insert_generated_question(
        self,
        source_question_id: str,
        perseus_json: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Insert a generated question into the output collection."""
        try:
            doc = {
                "source_question_id": ObjectId(source_question_id),
                "perseus_json": perseus_json,
                "metadata": {
                    "generated_at": datetime.utcnow(),
                    "llm_model": metadata.get("llm_model", "gemini-2.0-flash") if metadata else "gemini-2.0-flash",
                    "validation_status": metadata.get("validation_status", "valid") if metadata else "valid",
                    "attempt_count": metadata.get("attempt_count", 1) if metadata else 1,
                },
                "skills": metadata.get("skills", []) if metadata else [],
                "subject": metadata.get("subject", "unknown") if metadata else "unknown",
                "widget_types": self._extract_widget_types(perseus_json),
            }

            result = self.generated.insert_one(doc)
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Error inserting generated question: {e}")
            return None

    def _extract_widget_types(self, perseus_json: dict[str, Any]) -> list[str]:
        """Extract widget types from Perseus JSON."""
        widget_types = set()

        question = perseus_json.get("question", {})
        widgets = question.get("widgets", {})

        for widget_data in widgets.values():
            widget_type = widget_data.get("type")
            if widget_type:
                widget_types.add(widget_type)

        return list(widget_types)

    def get_widget_type_stats(self) -> dict[str, int]:
        """Get widget type distribution from source collection."""
        try:
            pipeline = [
                {"$project": {"widgets": {"$objectToArray": "$question.widgets"}}},
                {"$unwind": "$widgets"},
                {"$group": {"_id": "$widgets.v.type", "count": {"$sum": 1}}},
                {"$sort": {"count": DESCENDING}},
            ]

            result = self.source.aggregate(pipeline)
            return {doc["_id"]: doc["count"] for doc in result if doc["_id"]}

        except Exception as e:
            logger.error(f"Error getting widget type stats: {e}")
            return {}

    def count_generated(self) -> int:
        """Count total generated questions."""
        try:
            return self.generated.count_documents({})
        except Exception as e:
            logger.error(f"Error counting generated questions: {e}")
            return 0

    def count_source(self) -> int:
        """Count total source questions."""
        try:
            return self.source.count_documents({})
        except Exception as e:
            logger.error(f"Error counting source questions: {e}")
            return 0


# Global repository instance
question_repo = QuestionRepository()
