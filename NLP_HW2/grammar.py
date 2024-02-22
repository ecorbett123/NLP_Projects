"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # First check that all the probabilities sum to one
        for entry in self.lhs_to_rules:
            total_probability = 0
            if len(self.lhs_to_rules[entry]) < 1:
                return False # no values provided for a token (probability not summing to one)
            for rule in self.lhs_to_rules[entry]:
                if len(rule) == 3:
                    total_probability += rule[2]
                else:
                    return False # rule not in correct format
            if not math.isclose(total_probability, 1.0):
                return False # probability for entry not summing to one

        # Next check that each rule is in Chomsky Normal form
        nonterminals = self.lhs_to_rules.keys()
        for entry in self.rhs_to_rules:
            if len(entry) != 2 and len(entry) != 1:
                return False # not two non-terminals or one terminal
            if len(entry) == 2:
                if entry[0] not in nonterminals or entry[1] not in nonterminals:
                    return False # should be tuple of two non-terminals
            if len(entry) == 1:
                if entry[0] in nonterminals:
                    return False # if length is one, should be just a terminal
        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        
