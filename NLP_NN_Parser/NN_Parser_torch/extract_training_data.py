from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence: 
            self.buffer = list(reversed(sentence))
        self.deps = set() 
    
    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)

   

def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label) 
        elif rel == "right_arc":
            state.right_arc(label) 
         
    return state.deps
   
class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None    
    def __repr__(self):
        return "<ROOT>"

     
def get_training_instances(dep_structure):

    deprels = dep_structure.deprels
    
    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1
 
    seq = []
    while state.buffer: 
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy() 
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else: 
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq   


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):
       
    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)        
        self.pos_vocab = self.read_vocab(pos_vocab_file)        
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))
    
        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file: 
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab     

    def get_input_representation(self, words, pos, state):
        state_rep = [0]*6
        i = len(state.stack)
        if i < 1:
            state_rep[0] = self.word_vocab["<NULL>"]
            state_rep[1] = self.word_vocab["<NULL>"]
            state_rep[2] = self.word_vocab["<NULL>"]
        elif i == 1:
            state_rep[0] = self.get_word_index(state.stack[-1], words, pos)
            state_rep[1] = self.word_vocab["<NULL>"]
            state_rep[2] = self.word_vocab["<NULL>"]
        elif i == 2:
            state_rep[0] = self.get_word_index(state.stack[-1], words, pos)
            state_rep[1] = self.get_word_index(state.stack[-2], words, pos)
            state_rep[2] = self.word_vocab["<NULL>"]
        else:
            state_rep[0] = self.get_word_index(state.stack[-1], words, pos)
            state_rep[1] = self.get_word_index(state.stack[-2], words, pos)
            state_rep[2] = self.get_word_index(state.stack[-3], words, pos)

        i = len(state.buffer)
        if i < 1:
            state_rep[3] = self.word_vocab["<NULL>"]
            state_rep[4] = self.word_vocab["<NULL>"]
            state_rep[5] = self.word_vocab["<NULL>"]
        elif i == 1:
            state_rep[3] = self.get_word_index(state.buffer[-1], words, pos)
            state_rep[4] = self.word_vocab["<NULL>"]
            state_rep[5] = self.word_vocab["<NULL>"]
        elif i == 2:
            state_rep[3] = self.get_word_index(state.buffer[-1], words, pos)
            state_rep[4] = self.get_word_index(state.buffer[-2], words, pos)
            state_rep[5] = self.word_vocab["<NULL>"]
        else:
            state_rep[3] = self.get_word_index(state.buffer[-1], words, pos)
            state_rep[4] = self.get_word_index(state.buffer[-2], words, pos)
            state_rep[5] = self.get_word_index(state.buffer[-3], words, pos)

        return np.array(state_rep)

    def get_word_index(self, word_index, words, pos):
        if word_index == 0:
            return self.word_vocab['<ROOT>']

        word = words[word_index]
        if word in self.word_vocab:
            return self.word_vocab[word]

        tag_pos = pos[word_index]
        if tag_pos == 'CD':
            return self.word_vocab['<CD>']
        if tag_pos == 'NNP' or tag_pos == 'NNPS':
            return self.word_vocab['<NNP>']

        return self.word_vocab['<UNK>']

    def get_output_representation(self, output_pair):
        one_hot = [0.0]*91
        one_hot[self.output_labels[output_pair]] = 1.0
        return np.array(one_hot)

     
    
def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0 
    for dtree in conll_reader(in_file): 
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)
       


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 


    with open(sys.argv[1],'r') as in_file:   

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)


