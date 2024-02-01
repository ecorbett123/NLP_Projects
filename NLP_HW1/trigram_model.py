import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    sequence_copy = sequence.copy()
    sequence_copy.append('STOP')
    initial_list = []
    n_gram_list = []
    seq_len = len(sequence_copy)

    initial_list.extend([sequence_copy[0:0 + i] for i in range(1, min(n, seq_len+1), 1)])

    # pad lists with 'START'
    for n_gram in initial_list:
        num_starts = n - len(n_gram)
        start_list = ['START'] * num_starts
        n_gram_list.append(start_list + n_gram)

    if n == 1:
        n_gram_list.append(['START'])

    if seq_len >= n:
        n_gram_list.extend([sequence_copy[i:i+n] for i in range(0, seq_len-n+1, 1)])

    n_gram_list = [tuple(x) for x in n_gram_list]
    return n_gram_list


class TrigramModel(object):
    
    def __init__(self, corpusfile):

        self.num_words_in_corpus = 0
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.lexicon_size = len(self.lexicon)

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        for line in corpus:
            uni_gram_list = get_ngrams(line, 1)
            for uni_gram in uni_gram_list:
                count = 0
                if uni_gram in self.unigramcounts.keys():
                    count = self.unigramcounts[uni_gram]
                self.unigramcounts[uni_gram] = count + 1
                # keep track of total number of words
                self.num_words_in_corpus += 1

            bi_gram_list = get_ngrams(line, 2)
            for bi_gram in bi_gram_list:
                count = 0
                if bi_gram in self.bigramcounts.keys():
                    count = self.bigramcounts[bi_gram]
                self.bigramcounts[bi_gram] = count + 1

            tri_gram_list = get_ngrams(line, 3)
            for tri_gram in tri_gram_list:
                count = 0
                if tri_gram in self.trigramcounts.keys():
                    count = self.trigramcounts[tri_gram]
                self.trigramcounts[tri_gram] = count + 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        u_v = (trigram[0], trigram[1])
        count_trigram = 0
        if trigram in self.trigramcounts.keys():
            count_trigram = self.trigramcounts[trigram]

        if u_v not in self.bigramcounts.keys():
            return 1/self.lexicon_size

        return count_trigram / self.bigramcounts[u_v]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        u = (bigram[0],)

        if bigram not in self.bigramcounts.keys():
            return 0.0

        # default count for u is unknown in case passed a word not in the lexicon
        count_u = self.unigramcounts[('UNK',)]
        if u in self.unigramcounts.keys():
            count_u = self.unigramcounts[u]

        return self.bigramcounts[bigram]/count_u
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if unigram not in self.unigramcounts.keys():
            return 0.0
        return self.unigramcounts[unigram]/self.num_words_in_corpus

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = []
        prev_two_tokens = ('START', 'START')

        next_token = ''
        while next_token != 'STOP' and len(result) < t:
            trigram_prob_map = {}
            total_prob = 0
            for key in self.trigramcounts.keys():
                if key[0] == prev_two_tokens[0] and key[1] == prev_two_tokens[1] and key[2] != 'UNK':
                    raw_tri_prob = self.raw_trigram_probability(key)
                    trigram_prob_map[key] = raw_tri_prob
                    total_prob += raw_tri_prob

            r = random.random()*total_prob
            cur_total_prob = 0

            if total_prob > 0:
                for key in trigram_prob_map:
                    cur_total_prob += trigram_prob_map[key]
                    if cur_total_prob >= r:
                        next_token = key[2]
                        result.append(next_token)
                        prev_two_tokens = (prev_two_tokens[1], next_token)
                        break
            else:
                # TODO: if i had more time, this would be handled more elegantly, perhaps by looking at bigram probs next
                # if two word sequence has never appeared, the next word is chosen randomly
                l = random.random()*self.lexicon_size
                for word in self.lexicon:
                    cur_total_prob += 1
                    if cur_total_prob >= l:
                        next_token = word
                        result.append(next_token)
                        prev_two_tokens = (prev_two_tokens[1], next_token)
                        break
        if len(result) >= t:
            result.append('STOP')
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        bigram = (trigram[1], trigram[2])
        unigram = (trigram[2], )

        return lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(bigram) + lambda3*self.raw_unigram_probability(unigram)
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        ngrams = get_ngrams(sentence, 3)
        total_prob = 0
        for trigram in ngrams:
            smoothed_trigram = self.smoothed_trigram_probability(trigram)
            if smoothed_trigram != 0:
                total_prob += math.log2(smoothed_trigram)
            else:
                return 0 # this should never happen because unigram always has non-zero probability
        return float(total_prob)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        # M = num words in the corpus
        num_tokens = 0
        total_probs = 0
        for line in corpus:
            num_tokens += len(line)
            total_probs += self.sentence_logprob(line)
        return float(2**(-1*(total_probs/num_tokens)))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        # high score directory
        for f in os.listdir(testdir1):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            if pp_high < pp_low:
                correct += 1
            total += 1

        # low score directory
        for f in os.listdir(testdir2):
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))

            if pp_low < pp_high:
                correct += 1
            total += 1
        
        return float(correct/total)*100

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

