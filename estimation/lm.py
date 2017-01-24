from math import log, exp
from collections import defaultdict, Counter
from zipfile import ZipFile
import re
# additional imports
from numpy.random import multinomial

kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

kWORDS = re.compile("[a-z]{1,}")
kREP = set(["Bush", "GWBush", "Eisenhower", "Ford", "Nixon", "Reagan"])
kDEM = set(["Carter", "Clinton", "Truman", "Johnson", "Kennedy"])

class OutOfVocab(Exception):
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)

def sentences_from_zipfile(zip_file, filter_presidents):
    """
    Given a zip file, yield an iterator over the lines in each file in the
    zip file.
    """
    with ZipFile(zip_file) as z:
        for ii in z.namelist():
            try:
                pres = ii.replace(".txt", "").replace("state_union/", "").split("-")[1]
            except IndexError:
                continue

            if pres in filter_presidents:
                for jj in z.read(ii).decode(errors='replace').split("\n")[3:]:
                    yield jj.lower()

def tokenize(sentence):
    """
    Given a sentence, return a list of all the words in the sentence.
    """
    
    return kWORDS.findall(sentence.lower())

def bigrams(sentence):
    """
    Given a sentence, generate all bigrams in the sentence.
    """
    
    for ii, ww in enumerate(sentence[:-1]):
        yield ww, sentence[ii + 1]




class BigramLanguageModel:

    def __init__(self):
        self._vocab = set([kSTART, kEND])
        
        # Add your code here!
        # Bigram counts
        self._counts = dict()
        self._pvals = []
        self._associations = []
        self._pvals_stored = False
        self._vocab_final = False

    def train_seen(self, word):
        """
        Tells the language model that a word has been seen.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here!
        self._vocab.add(word)

    def generate(self, context):
        """
        Given the previous word of a context, generate a next word from its
        conditional language model probability.
        """

        # Add your code here.  Make sure to the account for the case
        # of a context you haven't seen before and Don't forget the
        # smoothing "+1" term while sampling.

        # Your code here
        n = len(self._counts.values())
        if not self._pvals_stored:
            self.store_pvals(context)
        # intensely convoluted...
        # Do a single sample from multinomial distribution, then use that index to
        # find the corresponding word
        sample = multinomial(1, self._pvals)
        index = list(sample).index(1)
        word = self._associations[index]
        return word
    
    # Called by Generate(), produces the appropriate array for multinomial function
    # takes a really long time (5+ min) when run for the first time
    def store_pvals(self, context):
        # print("Generating pval total...") # helpful for the wait
        for word in self._vocab:
            prob = exp(self.laplace(context, word)) # self.laplace returns log probs
            self._pvals.append(prob)
            self._associations.append(word) # store word at the same index
        pval_total = sum(self._pvals)
        # print("Done!")

        i = 0
        # print("Storing pvals...")
        for pval in self._pvals:
            self._pvals[i] /= pval_total
            i += 1

        self._pvals_stored = True
        # print("Done!")

    def sample(self, sample_size):
        """
        Generate an English-like string from a language model of a specified
        length (plus start and end tags).
        """

        # You should not need to modify this function
        yield kSTART
        next = kSTART
        for ii in range(sample_size):
            next = self.generate(next)
            if next == kEND:
                break
            else:
                yield next
        yield kEND
            
    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """
        
        # you should not need to modify this function
        
        self._vocab_final = True

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or testing.
        Prefix the sentence with <s>, generate the words in the
        sentence, and end the sentence with </s>.
        """

        # you should not need to modify this function
        
        yield kSTART
        for ii in tokenize(sentence):
            if ii not in self._vocab:
                raise OutOfVocab(ii)
            yield ii
        yield kEND

    def vocab(self):
        """
        Returns the language model's vocabulary
        """

        assert self._vocab_final, "Vocab not finalized"
        return list(sorted(self._vocab))
        
    def laplace(self, context, word):
        """
        Return the log probability (base e) of a word given its context
        """

        assert context in self._vocab, "%s not in vocab" % context
        assert word in self._vocab, "%s not in vocab" % word

        # Add your code here
        bigram_count = self._counts.get((context, word), 0) + 1 # laplace smoothing
        unigram_count = self.context_count(context) # laplace smoothing
        val = bigram_count / float(unigram_count + len(self._vocab)) # avoid int division
        return log(val)

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """

        # You'll need to complete this function, but here's a line of code that
        # will hopefully get you started.
        for context, word in bigrams(list(self.tokenize_and_censor(sentence))):
            assert word in self._vocab, "%s not in vocab" % word
            self._counts[(context, word)] = self._counts.get((context, word), 0) + 1

    def log_likelihood(self, sentence):
        """
        Compute the log likelihood of a sentence, divided by the number of
        tokens in the sentence.
        """
        prob = 0.0
        for context, word in bigrams(list(self.tokenize_and_censor(sentence))):
            prob += self.laplace(context, word)
        return prob / len(self._counts.keys())

    # Helper Functions:

    # finds the single counts of a given word as context
    def context_count(self, word):
        sum = 0
        for key in self._counts:
            if word in key[0]:
                sum += 1
        return sum

if __name__ == "__main__":
    dem_lm = BigramLanguageModel()
    rep_lm = BigramLanguageModel()

    for target, pres, name in [(dem_lm, kDEM, "D"), (rep_lm, kREP, "R")]:
        for sent in sentences_from_zipfile("../data/state_union.zip", pres):
            for ww in tokenize(sent):
                target.train_seen(ww)
                
        print("Done looking at %s words, finalizing vocabulary" % name)
        target.finalize()
        
        for sent in sentences_from_zipfile("../data/state_union.zip", pres):
            target.add_train(sent)
    
        print("Trained language model for %s" % name)

    # Sentence generation
    # i = 0
    # while i < 10:
    #     dem_string = "dem: "
    #     rep_string = "rep: "
    #     dem_sentence = dem_lm.sample(20)
    #     rep_sentence = rep_lm.sample(20)
    #     for word in dem_sentence:
    #         dem_string = "%s %s" % (dem_string, word)
    #     for word in rep_sentence:
    #         rep_string = "%s %s" % (rep_string, word)
    #     print(dem_string)
    #     print(rep_string)
    #     i += 1

    with open("../data/2016-obama.txt") as infile:
        print("REP\t\tDEM\t\tSentence\n" + "=" * 80)
        for ii in infile:
            if len(ii) < 15: # Ignore short sentences
                continue
            try:
                dem_score = dem_lm.log_likelihood(ii)
                rep_score = rep_lm.log_likelihood(ii)

                print("%f\t%f\t%s" % (dem_score, rep_score, ii.strip()))
            except OutOfVocab:
                None