"""
Context-aware spell correction for Shan language.

This module provides sentence-level spell correction that uses surrounding
words to make better correction decisions. Optimized for real-time typing
assistance with <100ms latency.

Supports two reranking modes:
1. N-gram based: Uses bigram/trigram probabilities for context
2. Neural based: Uses trained neural reranker for better accuracy
"""

import time
from typing import List, Tuple, Optional, Dict, Union
from shannlp.tokenize import word_tokenize
from shannlp.spell.core import spell_correct, SpellCorrector
from shannlp.spell.ngram import NgramModel

# Optional neural model import
try:
    from shannlp.spell.neural import SpellReranker
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    SpellReranker = None


class ContextAwareCorrector:
    """
    Context-aware spell corrector using n-gram language model.

    Combines word-level spell correction with context probabilities
    from surrounding words for improved accuracy.

    Examples:
        >>> corrector = ContextAwareCorrector()
        >>> corrector.load_model("bigram_model.pkl")
        >>> result = corrector.correct_text("မိူင်း ယူႇ")
        >>> len(result) == 2
        True
    """

    def __init__(
        self,
        ngram_model: Optional[NgramModel] = None,
        spell_corrector: Optional[SpellCorrector] = None,
        context_window: int = 2,
        context_weight: float = 0.3,
        always_suggest_alternatives: bool = False,
        neural_model: Optional['SpellReranker'] = None,
        use_neural: bool = False
    ):
        """
        Initialize context-aware corrector.

        Args:
            ngram_model: Pre-trained n-gram model
            spell_corrector: Spell corrector instance
            context_window: Number of words to consider for context (1-2 recommended for speed)
            context_weight: Weight for context probability (0-1)
                           Final score = (1-w)*edit_score + w*context_score
            always_suggest_alternatives: If True, generate alternatives even for
                dictionary words (for neural reranking). Default False for speed.
            neural_model: Pre-trained neural reranker model (optional)
            use_neural: If True, use neural reranking (requires neural_model)
        """
        self.ngram_model = ngram_model
        self.spell_corrector = spell_corrector or SpellCorrector()
        self.context_window = min(context_window, 2)  # Limit for real-time performance
        self.context_weight = context_weight
        self.neural_model = neural_model
        self.use_neural = use_neural and neural_model is not None

        # Enable alternatives for dictionary words if using neural model
        if self.use_neural:
            self.always_suggest_alternatives = True
        else:
            self.always_suggest_alternatives = always_suggest_alternatives

        # Performance tracking
        self.correction_times: List[float] = []

    def load_model(self, model_path: str):
        """
        Load pre-trained n-gram model.

        Args:
            model_path: Path to saved model file

        Examples:
            >>> corrector = ContextAwareCorrector()
            >>> corrector.load_model("bigram_model.pkl")
        """
        self.ngram_model = NgramModel.load(model_path)
        print(f"N-gram model loaded ({self.ngram_model.n}-gram)")

    def load_neural_model(self, model_path: str, device: str = None):
        """
        Load pre-trained neural reranker model.

        Args:
            model_path: Path to saved neural model (.pt file)
            device: Device to use (cuda, mps, cpu). Auto-detected if None.

        Examples:
            >>> corrector = ContextAwareCorrector()
            >>> corrector.load_neural_model("spell_reranker.pt")
        """
        if not NEURAL_AVAILABLE:
            raise ImportError(
                "Neural model support requires PyTorch. "
                "Install with: pip install torch"
            )

        from shannlp.spell.neural import SpellReranker
        self.neural_model = SpellReranker.load(model_path, device)
        self.use_neural = True
        self.always_suggest_alternatives = True  # Required for neural reranking
        print(f"Neural reranker loaded from {model_path}")

    def correct_text(
        self,
        text: str,
        max_suggestions: int = 1,
        min_confidence: float = 0.3
    ) -> List[str]:
        """
        Correct spelling in text using context.

        Args:
            text: Input text to correct
            max_suggestions: Maximum suggestions per word (1 for real-time)
            min_confidence: Minimum confidence threshold

        Returns:
            List of corrected tokens

        Examples:
            >>> corrector = ContextAwareCorrector()
            >>> result = corrector.correct_text("မိူင်း ယူႇ")
            >>> len(result) > 0
            True
        """
        start_time = time.time()

        # Tokenize
        tokens = word_tokenize(text, keep_whitespace=False)
        if not tokens:
            return []

        # Correct each token with context
        corrected_tokens = []
        for idx, token in enumerate(tokens):
            # Skip if already correct AND we don't want alternatives
            # (if always_suggest_alternatives is True, we still want to
            # consider alternatives for dictionary words based on context)
            if self.spell_corrector.is_correct(token) and not self.always_suggest_alternatives:
                corrected_tokens.append(token)
                continue

            # Get context
            context_before = tokens[max(0, idx - self.context_window):idx]
            context_after = tokens[idx + 1:min(len(tokens), idx + self.context_window + 1)]

            # Get correction with context
            correction = self._correct_with_context(
                token,
                context_before,
                context_after,
                max_suggestions,
                min_confidence
            )

            corrected_tokens.append(correction)

        # Track performance
        elapsed = time.time() - start_time
        self.correction_times.append(elapsed)

        return corrected_tokens

    def _correct_with_context(
        self,
        word: str,
        context_before: List[str],
        context_after: List[str],
        max_suggestions: int,
        min_confidence: float
    ) -> str:
        """
        Correct single word using surrounding context.

        Args:
            word: Word to correct
            context_before: Words before target
            context_after: Words after target
            max_suggestions: Maximum suggestions
            min_confidence: Minimum confidence

        Returns:
            Corrected word (or original if no good correction found)
        """
        # Get spell correction candidates
        candidates = spell_correct(
            word,
            max_edit_distance=2,
            max_suggestions=max_suggestions * 3,  # Get more for re-ranking
            use_frequency=True,
            use_phonetic=True,
            min_confidence=0.1,  # Lower threshold, will re-rank
            always_suggest_alternatives=self.always_suggest_alternatives
        )

        if not candidates:
            return word

        # Use neural reranking if available
        if self.use_neural and self.neural_model is not None:
            return self._rerank_with_neural(
                word,
                candidates,
                context_before,
                context_after,
                min_confidence
            )

        # If no n-gram model, use top candidate from spell corrector
        if self.ngram_model is None or not self.ngram_model.is_trained:
            top_candidate, confidence = candidates[0]
            return top_candidate if confidence >= min_confidence else word

        # Re-rank using n-gram context
        reranked = self._rerank_with_context(
            candidates,
            context_before,
            context_after
        )

        if reranked:
            best_candidate, final_score = reranked[0]
            if final_score >= min_confidence:
                return best_candidate

        return word

    def _rerank_with_neural(
        self,
        word: str,
        candidates: List[Tuple[str, float]],
        context_before: List[str],
        context_after: List[str],
        min_confidence: float
    ) -> str:
        """
        Re-rank candidates using neural model.

        Args:
            word: Original word
            candidates: List of (candidate, edit_score) tuples
            context_before: Words before target
            context_after: Words after target
            min_confidence: Minimum confidence threshold

        Returns:
            Best candidate word
        """
        # Extract candidate words
        candidate_words = [c[0] for c in candidates]

        # Build context strings
        context_left = " ".join(context_before)
        context_right = " ".join(context_after)

        # Get neural prediction
        best_candidate, scores = self.neural_model.predict(
            word,
            candidate_words,
            context_left,
            context_right
        )

        return best_candidate

    def _rerank_with_context(
        self,
        candidates: List[Tuple[str, float]],
        context_before: List[str],
        context_after: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates using context probability from n-gram model.

        Args:
            candidates: List of (word, edit_score) tuples
            context_before: Context words before
            context_after: Context words after

        Returns:
            Re-ranked list of (word, combined_score) tuples
        """
        scored = []

        for candidate_word, edit_score in candidates:
            # Calculate context score
            context_score = 0.0

            # Score with previous context
            if context_before and self.ngram_model:
                try:
                    prob_before = self.ngram_model.probability(
                        candidate_word,
                        context_before[-self.context_window:]
                    )
                    context_score += math.log(prob_before) if prob_before > 0 else -10
                except:
                    context_score = 0.0

            # Score with following context (if available and using trigrams)
            if context_after and self.ngram_model and self.ngram_model.n >= 3:
                try:
                    # P(next_word | candidate, prev)
                    next_word = context_after[0]
                    context_for_next = (context_before + [candidate_word])[-self.context_window:]
                    prob_after = self.ngram_model.probability(next_word, context_for_next)
                    context_score += math.log(prob_after) if prob_after > 0 else -10
                except:
                    pass

            # Normalize context score to 0-1 range (rough approximation)
            # Context scores are negative log probs, typically -20 to 0
            normalized_context = min(1.0, max(0.0, (context_score + 20) / 20))

            # Combined score
            final_score = (
                (1 - self.context_weight) * edit_score +
                self.context_weight * normalized_context
            )

            scored.append((candidate_word, final_score))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def correct_sentence(
        self,
        sentence: str,
        return_original_if_no_change: bool = True
    ) -> str:
        """
        Correct spelling in a sentence.

        Args:
            sentence: Input sentence
            return_original_if_no_change: Return original if no corrections made

        Returns:
            Corrected sentence as string

        Examples:
            >>> corrector = ContextAwareCorrector()
            >>> result = corrector.correct_sentence("မိူင်း ယူႇ")
            >>> len(result) > 0
            True
        """
        corrected_tokens = self.correct_text(sentence)

        if not corrected_tokens:
            return sentence if return_original_if_no_change else ""

        # Join tokens (Shan typically has no spaces, but respect tokenization)
        return " ".join(corrected_tokens)

    def batch_correct(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[str]]:
        """
        Correct multiple texts in batch.

        Args:
            texts: List of texts to correct
            show_progress: Show progress indicator

        Returns:
            List of corrected token lists

        Examples:
            >>> corrector = ContextAwareCorrector()
            >>> results = corrector.batch_correct(["text1", "text2"])
            >>> len(results) == 2
            True
        """
        results = []

        for i, text in enumerate(texts):
            if show_progress and i % 100 == 0:
                print(f"Processing {i}/{len(texts)}...")

            corrected = self.correct_text(text)
            results.append(corrected)

        return results

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics

        Examples:
            >>> corrector = ContextAwareCorrector()
            >>> corrector.correct_text("test")
            >>> stats = corrector.get_performance_stats()
            >>> 'avg_time_ms' in stats
            True
        """
        if not self.correction_times:
            return {
                'avg_time_ms': 0.0,
                'max_time_ms': 0.0,
                'min_time_ms': 0.0,
                'total_corrections': 0
            }

        times_ms = [t * 1000 for t in self.correction_times]

        stats = {
            'avg_time_ms': sum(times_ms) / len(times_ms),
            'max_time_ms': max(times_ms),
            'min_time_ms': min(times_ms),
            'total_corrections': len(self.correction_times)
        }

        # Add n-gram model cache stats if available
        if self.ngram_model:
            cache_stats = self.ngram_model.get_cache_stats()
            stats.update(cache_stats)

        return stats


# Import math for log operations
import math
