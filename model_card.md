# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:** 
This project compares **both** a rule-based model and a machine learning model.

**Intended purpose:** 
To classify short, informal text messages (like social media updates or texts) into one of four mood categories: `positive`, `negative`, `neutral`, or `mixed`.

**How it works (brief):** 
* **Rule-Based:** Tokenizes text, checks for basic negation, and cross-references words against predefined `POSITIVE_WORDS` and `NEGATIVE_WORDS` lists. It tracks positive and negative matches separately to allow for "mixed" emotion detection.
* **ML Model:** Uses a "Bag of Words" approach (`CountVectorizer`) to convert text into numeric arrays, then trains a Logistic Regression model to find mathematical correlations between specific vocabulary words and the provided labels.

## 2. Data

**Dataset description:** 
The original `SAMPLE_POSTS` dataset contained 6 simple sentences. It was expanded to 14 posts to better reflect realistic internet language. 

**Labeling process:** 
Labels (`positive`, `negative`, `neutral`, `mixed`) were applied manually based on human interpretation of the sentence's primary intent. 

**Important characteristics of your dataset:** 
The expanded dataset intentionally includes:
* **Slang:** "mid", "no cap", "immaculate", "highkey"
* **Emojis:** ✨, 💀, 🥳, 🙌, 🙄
* **Sarcasm:** "I absolutely love it when my computer crashes right before I save my work"
* **Mixed Feelings:** "Lowkey failing this project but no cap the vibes are immaculate"

**Possible issues with the dataset:** 
The dataset is microscopic (14 items) and highly imbalanced. The labels are also highly subjective; for instance, "Feeling pretty mid today" was labeled `neutral`, but some might argue "mid" leans `negative`. 

## 3. How the Rule-Based Model Works

**Your scoring rules:**
The model strips punctuation from the edges of words (preserving emojis). It then loops through the tokens:
* **Scoring:** +1 to `pos_score` for words in the positive list, +1 to `neg_score` for words in the negative list. 
* **Enhancement (Negation):** If a matched word is immediately preceded by a negation word (e.g., "not", "never"), the logic swaps the boolean flag (a positive word is counted as a negative hit, and vice versa).
* **Enhancement (Mixed Emotions):** By tracking scores separately rather than keeping a single running integer, the model successfully returns `mixed` if both `pos_score > 0` and `neg_score > 0`.

**Strengths of this approach:** 
It is incredibly fast, completely transparent (you know exactly why it made a prediction), and handles explicit mixed emotions well. 

**Weaknesses of this approach:** 
It is entirely dictionary-dependent. If a word or emoji isn't in the hardcoded lists, the model is completely blind to it. It also possesses zero contextual awareness, making it highly susceptible to sarcasm.

## 4. How the ML Model Works 

**Features used:** 
Bag of words using `CountVectorizer`.

**Training data:** 
Trained on the exact same 14 strings in `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior & Comparison:** 
The ML model achieved 100% accuracy on the dataset, vastly outperforming the initial rule-based model. It appeared to perfectly "understand" the sarcasm and slang without needing dictionary updates. However, this is because it perfectly memorized the training data (overfitting). 
* *Differences:* The ML model "fixed" the sarcasm failure because it associated the word "crashes" with the `negative` label automatically based on the training data. 
* *Sensitivity:* The model is highly sensitive to the exact vocabulary it was trained on. If tested with a synonym it hasn't seen (e.g., "thrilled" instead of "excited"), it will fail.

## 5. Evaluation

**How you evaluated the model:** 
Evaluated by iterating over `SAMPLE_POSTS` and comparing the predicted label against `TRUE_LABELS`. 

**Examples of correct predictions (Rule-Based):** 
* *"Feeling tired but kind of hopeful"* -> `mixed`. (Correct because it caught "tired" as negative and "hopeful" as positive, firing the mixed threshold).
* *"Best day ever! 🥳🙌"* -> `positive`. (Correct, but only after expanding the dictionary to explicitly include "best", "🥳", and "🙌").

**Examples of incorrect predictions (Rule-Based):** 
* *"I absolutely love it when my computer crashes..."* -> predicted=`positive`, true=`negative`. The model sees "love" (+1), ignores the rest of the context, and fails to recognize the sarcasm.
* *"Feeling pretty mid today, honestly"* -> predicted=`neutral` (0), true=`neutral`. While technically a correct prediction based on the labels, the model only got it right by accident; it scored 0 because it didn't know the word "mid", not because it actually understood the sentence was neutral.

## 6. Limitations

* **Sarcasm Blindness:** As shown in the "computer crashes" example, rule-based systems cannot detect tone or irony.
* **Brittle Vocabularies:** A user typing "I'm extremely joyous" will register as `neutral` simply because "joyous" wasn't hardcoded into the list. 
* **Complex Negation:** The model only looks one word backward for negation. A sentence like "I wouldn't exactly say I'm happy" will still be marked `positive` because "wouldn't" is too far away from "happy".
* **ML Overfitting:** The machine learning model cannot generalize. It only works on the exact 14 sentences it has already seen.

## 7. Ethical Considerations (Bias & Scope)

**Bias and Scope:** This specific implementation is heavily optimized for Gen-Z/Millennial, extremely-online English speakers. Words like "mid", "no cap", and "vibes" were explicitly catered to. 
* **Who it misinterprets:** It will likely fail on older demographics, formal text, non-native English speakers, or dialects like AAVE where words like "bad" or "sick" can be used as strong positive amplifiers. 
* **Real-world Impact:** If this model were used in a customer service setting to automatically close "positive" tickets, a user writing sarcastic complaints (e.g., "Great job losing my package!") would have their ticket wrongfully closed. More dangerously, a mental health application relying on simple dictionaries might classify a distressed, complex message as `neutral` and fail to route the user to human help.

## 8. Ideas for Improvement

* **Implement a Train/Test Split:** For the ML model, separate the data so it trains on 80% of the posts and is evaluated on 20% unseen posts to get a real accuracy metric.
* **Switch to TF-IDF:** Instead of simple word counts, use TF-IDF in the ML model to down-weight common filler words and emphasize unique emotional indicators.
* **Lemmatization:** Add a library like NLTK or spaCy to the rule-based preprocessor so words like "crashes", "crashed", and "crashing" all map to a single root word ("crash"), reducing the need for massive dictionary lists.
* **Contextual AI:** Graduate from Scikit-Learn to a pre-trained Transformer model (like BERT or RoBERTa) that inherently understands the context of surrounding words to better catch sarcasm.
