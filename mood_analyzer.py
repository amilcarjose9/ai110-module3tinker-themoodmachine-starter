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

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" with an enhancement: Negation Handling.
        If a word like "not" or "never" appears immediately before a mood word, 
        it flips the score of that word.
        """
        tokens = self.preprocess(text)
        score = 0
        
        # Simple set of negation markers
        negation_words = {"not", "no", "never", "none", "isn't", "aren't", "wasn't", "don't", "doesn't"}

        for i, token in enumerate(tokens):
            token_score = 0
            
            # Base point value
            if token in self.positive_words:
                token_score = 1
            elif token in self.negative_words:
                token_score = -1
            
            # If we found a mood word, check the PREVIOUS token to see if it's a negation
            if token_score != 0 and i > 0 and tokens[i-1] in negation_words:
                token_score = -token_score # Flip the sign!

            score += token_score

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score into a mood label.
        """
        score = self.score_text(text)
        
        # Simple threshold mapping
        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
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
