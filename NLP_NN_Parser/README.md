# Dependency Parser with Neural Networks

## Overview
This project implements a feed-forward neural network to predict the transitions of an arc-standard dependency parser. The network takes in a representation of the current parser state and outputs a transition (shift, left_arc, right_arc) along with a dependency relation label.

Some implementation includes:
- Input representation for the neural network
- Decoding of the network's output
- Specification of the network architecture and training the model

The implementation is inspired by the paper:
**Chen, D., & Manning, C. (2014). "A fast and accurate dependency parser using neural networks."**

### Data Files
The dataset is a subset of the Penn Treebank, converted to dependency format:
- `data/train.conll` - Training set (~40k sentences)
- `data/dev.conll` - Development set (~5k sentences)
- `data/sec0.conll` - Small test set (~2k sentences)
- `data/test.conll` - Final test set (~2.5k sentences, do not use until final testing)

## Dependency Format
The data follows a modified CoNLL-X format, where each word has fields such as:
- Word ID, Form, POS tags, Head ID, Dependency relation, etc.

Example:
```
1  The    _ DT DT _ 2 det _ _
2  cat    _ NN NN _ 3 nsubj _ _
3  eats   _ VB VB _ 0 root _ _
4  tasty  _ JJ JJ _ 5 amod _ _
5  fish   _ NN NN _ 3 dobj _ _
6  .      _ .  .  _ 3 punct _ _
```

## Implementation Details

### Step 1: Obtain the Vocabulary
Run:
```sh
$ python get_vocab.py data/train.conll data/words.vocab data/pos.vocab
```
This creates indexed vocabulary files for words and POS tags.

### Step 2: Extract Input/Output Training Data
Generate training matrices:
```sh
$ python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
$ python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy
```

### Step 3: Train the Model
Run training:
```sh
$ python train_model.py
```
The training loop uses:
- `Adagrad` optimizer with learning rate `0.01`
- Cross-entropy loss function
- Mini-batches of size `16`
- 5 epochs of training

Performance:
- Training loss < `0.31`
- Training accuracy ~ `90%`

### Step 5: Parse and Evaluate
Parse sentences using the trained model:
```sh
$ python decoder.py data/sec0.conll
```
Evaluate parser performance:
```sh
$ python evaluate.py data/test.conll
```

## Summary
This project implements a dependency parser using a neural network. By extracting structured training data, designing an effective input representation, and training a feed-forward network, I am able to achieve high parsing accuracy.


