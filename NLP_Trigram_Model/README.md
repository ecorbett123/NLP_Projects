# Trigram Language Model

## Overview

This project implements a trigram language model in Python. The model is built by counting n-gram occurrences in a given corpus and then computing probabilities dynamically. It includes functionalities for text generation, probability estimation, sentence scoring, perplexity evaluation, and text classification.

## Features

- **N-gram Extraction:** Extracts n-grams from tokenized sentences with start and stop padding.
- **Corpus Processing:** Reads corpus files and replaces unseen words with an "UNK" token.
- **N-gram Counting:** Counts unigram, bigram, and trigram occurrences in the training corpus.
- **Probability Computation:** Computes raw and smoothed n-gram probabilities using linear interpolation.
- **Sentence Scoring:** Computes sentence log probabilities based on trigram probabilities.
- **Perplexity Calculation:** Evaluates model performance by computing perplexity on a test corpus.
- **Text Classification:** Uses perplexity-based classification for essay scoring.

## Usage

### 1. Extracting N-grams

```python
from trigram_model import get_ngrams
get_ngrams(["natural", "language", "processing"], 2)
```

**Output:**

```
[("START", "natural"), ("natural", "language"), ("language", "processing"), ("processing", "STOP")]
```

### 2. Training the Trigram Model

```python
from trigram_model import TrigramModel
model = TrigramModel("brown_train.txt")
```

### 3. Computing Raw Probabilities

```python
model.raw_trigram_probability(("START", "START", "the"))
model.raw_bigram_probability(("START", "the"))
model.raw_unigram_probability(("the",))
```

### 4. Generating Sentences

```python
model.generate_sentence()
```

**Example Output:**

```
["the", "last", "tread", ",", "mama", "did", "mention", "to", "the", "opposing", "sector", "of", "our", "natural", "resources", ".", "STOP"]
```

### 5. Sentence Log Probability

```python
sentence = ["the", "jury", "said", "the", "election", "was", "fair", "."]
log_prob = model.sentence_logprob(sentence)
```

### 6. Perplexity Calculation

```python
perplexity = model.perplexity("brown_test.txt")
print(f"Perplexity: {perplexity}")
```

### 7. Essay Scoring Experiment

```python
accuracy = model.essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Results

- **Perplexity:** Less than 400 on the Brown test corpus.
- **Essay Classification Accuracy:** 85%.

