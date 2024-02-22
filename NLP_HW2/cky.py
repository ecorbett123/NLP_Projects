"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        parse_table = {}
        num_tokens = len(tokens)
        # initialization
        for i in range(num_tokens):
            if (tokens[i],) not in self.grammar.rhs_to_rules:
                return False # terminal not in set of terminals
            parse_table[(i, i+1)] = self.grammar.rhs_to_rules[(tokens[i],)]

        for leng in range(2, num_tokens+1):
            for i in range(0, num_tokens - leng+1):
                j = i + leng
                for k in range(i+1, j):
                    # check each combination of nonterminals and add to i,j map index
                    if (i, k) in parse_table and (k, j) in parse_table:
                        replace_list = []
                        a_nonterminals = parse_table[(i, k)]
                        b_nonterminals = parse_table[(k, j)]
                        for a_nonterminal in a_nonterminals:
                            for b_nonterminal in b_nonterminals:
                                if (a_nonterminal[0], b_nonterminal[0]) in grammar.rhs_to_rules:
                                    replace_list.extend(grammar.rhs_to_rules[(a_nonterminal[0], b_nonterminal[0])])
                        if (i, j) not in parse_table and len(replace_list) > 0:
                            parse_table[i, j] = replace_list
                        elif len(replace_list) > 0:
                            parse_table[i, j].extend(replace_list)

        if (0, num_tokens) in parse_table:
            for entry in parse_table[(0, num_tokens)]:
                if entry[0] == 'TOP':
                    return True
        return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        table = {}
        probs = {}
        num_tokens = len(tokens)
        # initialization
        for i in range(num_tokens):
            if (tokens[i],) not in self.grammar.rhs_to_rules:
                return None, None  # terminal not in set of terminals

            table_map = {}
            prob_map = {}
            # pick the terminal with the highest probability for a given non-terminal
            for terminal in self.grammar.rhs_to_rules[(tokens[i],)]:
                if terminal[0] not in table_map or math.log2(terminal[2]) > prob_map[terminal[0]]:
                    table_map[terminal[0]] = tokens[i]
                    prob_map[terminal[0]] = math.log2(terminal[2])

            table[(i, i + 1)] = table_map
            probs[(i, i + 1)] = prob_map

        for leng in range(2, num_tokens + 1):
            for i in range(0, num_tokens - leng + 1):
                j = i + leng
                table_map = {}
                prob_map = {}
                for k in range(i + 1, j):
                    # check each combination of nonterminals and add max prob to table
                    if (i, k) in table and (k, j) in table:
                        a_nonterminals = table[(i, k)] # map of nonterminals to what made them up
                        b_nonterminals = table[(k, j)]
                        for a_nonterminal in a_nonterminals:
                            for b_nonterminal in b_nonterminals:
                                if (a_nonterminal, b_nonterminal) in grammar.rhs_to_rules:
                                    # for each of the rules, add them to map of greater probability
                                    for nonterminal in grammar.rhs_to_rules[(a_nonterminal, b_nonterminal)]:
                                        if nonterminal[0] not in table_map or math.log2(nonterminal[2]) + probs[(i,k)][a_nonterminal] + probs[(k,j)][b_nonterminal] > prob_map[nonterminal[0]]:
                                            table_map[nonterminal[0]] = ((a_nonterminal, i, k), (b_nonterminal, k, j))
                                            combined_prob = math.log2(nonterminal[2]) + probs[(i,k)][a_nonterminal] + probs[(k,j)][b_nonterminal]
                                            prob_map[nonterminal[0]] = combined_prob

                if len(table_map.keys()) > 0:
                    table[(i, j)] = table_map
                    probs[(i, j)] = prob_map

        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    return None 
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        #toks = ['she', 'saw', 'the', 'cat', 'with', 'glasses']
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        
