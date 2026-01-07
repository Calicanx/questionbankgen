"""Duplicate detection logic for generated questions."""

import re
import logging
from dataclasses import dataclass
from typing import Any, Optional, Set
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

@dataclass
class DuplicateResult:
    """Result of duplicate detection."""
    is_duplicate: bool
    confidence_score: float  # 0.0 to 1.0
    reason: str

class DuplicateDetector:
    """Detects if a generated question is a duplicate of its source."""

    def check_duplicate(
        self,
        source_json: dict[str, Any],
        generated_json: dict[str, Any]
    ) -> DuplicateResult:
        """
        Check if generated question is a duplicate of the source.
        
        Uses heuristics:
        1. Number set identity (if all numbers are the same, it's likely a duplicate)
        2. Text similarity (if text is >95% similar, it's a duplicate)
        """
        
        # Extract content
        source_content = source_json.get("question", {}).get("content", "")
        gen_content = generated_json.get("question", {}).get("content", "")
        
        # 1. Check text equality with whitespace normalization
        # This handles Markdown tables where spacing might differ slightly
        s_clean = " ".join(source_content.split())
        g_clean = " ".join(gen_content.split())
        
        logger.info(f"Checking Duplicate. Source: {s_clean[:100]}...")
        logger.info(f"Checking Duplicate. Gen:    {g_clean[:100]}...")

        if s_clean == g_clean:
            return DuplicateResult(True, 1.0, "Content matches (normalized whitespace)")
            
        # 2. Check number set identity
        source_nums = self._extract_numbers(source_content)
        gen_nums = self._extract_numbers(gen_content)
        
        if source_nums and gen_nums:
            if source_nums == gen_nums:
                # Same numbers found. Check text similarity.
                similarity = self._calculate_similarity(s_clean, g_clean)
                logger.info(f"Duplicate Check: Same numbers {source_nums}. Similarity: {similarity}")
                if similarity > 0.8: 
                    return DuplicateResult(
                        True, 
                        0.9 + (similarity * 0.1), 
                        f"Identical numbers {source_nums} and high text similarity ({similarity:.2f})"
                    )
        elif not source_nums and not gen_nums:
            # No numbers (conceptual question). Rely on high text similarity.
            similarity = self._calculate_similarity(s_clean, g_clean)
            logger.info(f"Duplicate Check: No numbers. Similarity: {similarity}")
            if similarity > 0.9:
                 return DuplicateResult(True, similarity, f"Text is extremely similar ({similarity:.2f})")

        # 3. Check for high conceptual similarity
        similarity = self._calculate_similarity(s_clean, g_clean)
        logger.info(f"Duplicate Check: Final Similarity: {similarity}")
        
        if similarity > 0.95:
            return DuplicateResult(True, similarity, f"Text is extremely similar ({similarity:.2f})")
            
        return DuplicateResult(False, similarity, "Content appears distinct")

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract all numbers from text, normalized."""
        # Match integers and decimals, ignoring LaTeX symbols like $ or {}
        # Simple regex: find digits, optionally with decimal point
        # We process matching strings to float to normalize "1" and "1.0"
        
        # Remove potential LaTeX confusion
        clean_text = re.sub(r'\$|\\|{|}|\[|\]', ' ', text)
        
        matches = re.findall(r'-?\d*\.?\d+', clean_text)
        numbers = set()
        for m in matches:
            if m == '.' or m == '-': continue
            try:
                # Store as float for comparison to handle 5 vs 5.0
                numbers.add(float(m))
            except ValueError:
                pass
        return numbers

    def _calculate_similarity(self, a: str, b: str) -> float:
        """Calculate normalized similarity ratio 0.0-1.0."""
        return SequenceMatcher(None, a, b).ratio()

# Global instance
duplicate_detector = DuplicateDetector()
