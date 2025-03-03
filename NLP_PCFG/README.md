# CKY Parser for Probabilistic Context-Free Grammars (PCFG)

## Overview

This project implements the CKY (Cocke-Kasami-Younger) algorithm for parsing sentences using a Probabilistic Context-Free Grammar (PCFG). The parser is capable of determining whether a sentence belongs to the language defined by a given grammar, retrieving the most probable parse tree, and evaluating its performance against a test corpus.

## Features

- **Grammar Processing:** Reads and verifies PCFG grammars in Chomsky Normal Form (CNF).
- **CKY Membership Checking:** Determines whether a given sentence can be parsed by the grammar.
- **Probabilistic Parsing:** Implements CKY parsing with backpointers to find the most probable parse tree.
- **Parse Tree Reconstruction:** Recovers full parse trees from the parse chart.
- **Performance Evaluation:** Computes precision, recall, and F-score for parsed sentences.

## Usage

### 1. Loading a PCFG Grammar

```python
from grammar import Pcfg
with open('atis3.pcfg', 'r') as grammar_file:
    grammar = Pcfg(grammar_file)
```

### 2. Verifying Grammar Validity

```python
grammar.verify_grammar()
```

### 3. Running CKY Membership Checking

```python
from cky import CkyParser
parser = CkyParser(grammar)
tokens = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
print(parser.is_in_language(tokens))  # True or False
```

### 4. Parsing with Backpointers

```python
table, probs = parser.parse_with_backpointers(tokens)
```

### 5. Retrieving a Parse Tree

```python
from cky import get_tree
parse_tree = get_tree(table, 0, len(tokens), grammar.startsymbol)
print(parse_tree)
```

### 6. Evaluating the Parser

```bash
python evaluate_parser.py atis3.pcfg atis3_test.ptb
```

## Results

- **Coverage:** ~67%
- **Average F-score (parsed sentences):** ~0.95
- **Average F-score (all sentences):** ~0.64
