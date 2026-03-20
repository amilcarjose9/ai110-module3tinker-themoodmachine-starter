# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens.
        Strips common punctuation from the edges of words so "happy!" matches "happy".
        """
        cleaned = text.strip().lower()
        raw_tokens = cleaned.split()
        
        tokens = []
        for token in raw_tokens:
            clean_token = token.strip('.,!?"\'()[]')
            if clean_token:
                tokens.append(clean_token)
                
        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> Tuple[int, int]:
        """
        Compute numeric mood scores.
        
        FIX: Now returns a tuple of (pos_score, neg_score) so mixed 
        emotions don't cancel each other out to zero.
        """
        tokens = self.preprocess(text)
        pos_score = 0
        neg_score = 0
        
        # Simple set of negation markers
        negation_words = {"not", "no", "never", "none", "isn't", "aren't", "wasn't", "don't", "doesn't"}

        for i, token in enumerate(tokens):
            is_positive = token in self.positive_words
            is_negative = token in self.negative_words
            
            # Negation handling: flips the boolean flag
            if (is_positive or is_negative) and i > 0 and tokens[i-1] in negation_words:
                is_positive, is_negative = is_negative, is_positive

            # Tally scores separately
            if is_positive:
                pos_score += 1
            if is_negative:
                neg_score += 1

        return pos_score, neg_score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score into a mood label.
        """
        pos_score, neg_score = self.score_text(text)
        
        # FIX: We can now accurately detect and return "mixed" labels
        if pos_score > 0 and neg_score > 0:
            return "mixed"
        elif pos_score > 0:
            return "positive"
        elif neg_score > 0:
            return "negative"
        else:
            # If both scores are exactly 0, no emotion words were found
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.
        """
        tokens = self.preprocess(text)
        positive_hits = [t for t in tokens if t in self.positive_words]
        negative_hits = [t for t in tokens if t in self.negative_words]
        score = self.score_text(text)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
