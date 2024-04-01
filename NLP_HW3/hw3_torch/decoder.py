import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def is_valid_action(self, action, stack, buffer):
        if action[0] == 'shift' and len(stack) != 0 and len(buffer) == 1:
            return False

        # arc-left or arc-right are not permitted if the stack is empty
        if action[0] != 'shift':
            if len(stack) < 1:
                return False

        # root node must never be the target of left-arc
        if action[0] == 'left_arc' and len(stack) == 1:
            return False

        return True

    def do_action(self, action, state):
        if action[0] == 'shift':
            state.shift()
        elif action[0] == 'left_arc': # left arc
            state.left_arc(action[1])
        elif action[0] == 'right_arc':
            state.right_arc(action[1])

    def parse_sentence(self, words, pos):

        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5
        while state.buffer:
            actions = self.model(torch.LongTensor(self.extractor.get_input_representation(words, pos, state)))
            indices = np.argsort(actions.tolist())[::-1]
            for index in indices:
                action = self.output_labels[index]
                if self.is_valid_action(action, state.stack, state.buffer):
                    self.do_action(action, state)
                    break

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
