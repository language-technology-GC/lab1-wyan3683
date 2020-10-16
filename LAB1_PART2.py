### LING83600: LANGUAGE TECHNOLOGY - LAB 1 PART 2
### by Winnie Yan

### Write-up at bottom 
'''
import nltk

# Assign variable to open news data
DATA = "news.2010.en.shuffled.deduped"
DATA = open(DATA, "r", encoding = "utf-8")

# Tokenize using NLTK word tokenizer and write to new .txt file
with open("news2010.txt", "w", encoding = "utf-8") as f:
    for line in DATA:
        line = line.casefold()
        line = nltk.word_tokenize(line)
        print(" ".join(line), file = f)
    print("Complete")
'''

# After tokenization complete, run provided PPMI script over tokenized data.
# Code below for calculating Spearman for 6 methods from Part 1
import logging
import pandas
import scipy.stats

from typing import List, Optional
from nltk.corpus import wordnet, wordnet_ic
from nltk.corpus.reader.wordnet import Synset


class Error(Exception):
    pass

class Calculate_Similarity:
    # Calculate similarity of word pairs
    
    # NLTK's pre-computed tables of information content
    def __init__(self, ic_path: str = "ic-brown.dat"):
        self.brown_ic = wordnet_ic.ic(ic_path)
    
    # Choosing first/supposedly most frequent lemma/synset
    def synset(lemma: str, pos: Optional[str] = None, lang: str = "eng") -> Synset:
        synsets = wordnet.synsets(lemma, pos, lang)
        return synsets[0]
    
    # Compute path similarity
    def path_sim(self, s1: Synset, s2: Synset) -> float:
        return s1.path_similarity(s2)
    
    # Compute Leacock-Chodorow similarity
    def lea_chod(self, s1: Synset, s2: Synset) -> float:
        return s1.lch_similarity(s2)
    
    # Compute Wu-Palmer similarity    
    def wu_palm(self, s1: Synset, s2: Synset) -> float:
        return s1.wup_similarity(s2)

    # Compute Resnik similarity
    def res(self, s1: Synset, s2: Synset) -> float:
        return s1.res_similarity(s2, self.brown_ic)
    
    # Compute Jiang-Conrath similarity
    def ji_con(self, s1: Synset, s2: Synset) -> float:
        return s1.jcn_similarity(s2, self.brown_ic)
    
    # Compute Line similarity
    def lin(self, s1: Synset, s2: Synset) -> float:
        return s1.lin_similarity(s2, self.brown_ic)
    
def spear_corr(x, y) -> float:
    # Calculate Spearman rho correlation
    return scipy.stats.spearmanr(x, y).correlation

def main():
    # Read data 
    data = pandas.read_csv("newsPPMIresults.tsv", delimiter = "\t")
        
    # Manually added missing column names to data set 
    
    # Casefold words in columns "first" and "second"
    data["first"] = data["first"].str.casefold()
    data["second"] = data["second"].str.casefold()
    
    # Make and append list of synset pairs
    ss_pairs: List[tuple[Synset, Synset]] = []
    for (x, word1, word2, score) in data.itertuples():
        x1 = Calculate_Similarity.synset(word1)
        x2 = Calculate_Similarity.synset(word2)
        ss_pairs.append((x1, x2))
    
    # Compute calculations for similarity scores 
    calc_sim = Calculate_Similarity()
    data["path_sim"] = [calc_sim.path_sim(x1, x2) for (x1, x2) in ss_pairs]
    data["lea_chod"] = [calc_sim.lea_chod(x1, x2) for (x1, x2) in ss_pairs]
    data["wu_palm"] = [calc_sim.wu_palm(x1, x2) for (x1, x2) in ss_pairs]
    data["res"] = [calc_sim.res(x1, x2) for (x1, x2) in ss_pairs]
    data["ji_con"] = [calc_sim.ji_con(x1, x2) for (x1, x2) in ss_pairs]
    data["lin"] = [calc_sim.lin(x1, x2) for (x1, x2) in ss_pairs] 
    
    # Compute correlations for similarity scores
    logging.info("path_sim\t%.4f", spear_corr(data["scale"], data["path_sim"]))
    logging.info("lea_chod\t%.4f", spear_corr(data["scale"], data["lea_chod"]))
    logging.info("wu_palm\t%.4f", spear_corr(data["scale"], data["wu_palm"]))
    logging.info("res\t%.4f", spear_corr(data["scale"], data["res"]))
    logging.info("ji_con\t%.4f", spear_corr(data["scale"], data["ji_con"]))
    logging.info("lin\t%.4f", spear_corr(data["scale"], data["lin"]))
    

if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    main()
    
    

# News Crawl from 2010
# INFO: 277 words tracked
# INFO: 202 pairs tracked
# INFO: 168029483 tokens counted
# INFO: 0 pairs covered
# Returned blank results

# News Crawl from 2008 
# INFO: 277 words tracked
# INFO: 202 pairs tracked
# INFO: 278831148 tokens counted
# INFO: 0 pairs covered

# REVISION: News Crawl from 2010
# INFO: 277 words tracked
# INFO: 203 pairs tracked
# INFO: 168019178 tokens counted
# INFO: 170 pairs covered


'''
>>> Spearman Correlation Coefficient (to 4 digits):
    INFO: path_sim	0.0769
    INFO: lea_chod	0.0769
    INFO: wu_palm	0.0535
    INFO: res	0.0362
    INFO: ji_con	-0.0967
    INFO: lin	-0.0173
    
>>> WORD PAIRS NOT COVERED (ERRORS AND RESOLUTIONS)
    (eat, drink)
    (stock, live)

>>> WRITE-UP (DESCRIPTION OF APPROACH & PROBLEMS ENCOUNTERED)
    
        For Part 2, my biggest concern was limited resources since the laptop
    I borrowed has lower memory and storage space than I am used to working 
    with. I chose to use the 2010 WMT news crawl data, as it had the smallest
    data size. To tokenize, I opened the text file as [data] and then wrote
    the tokenized lines into a new [.txt]. To ensure that the process had 
    completed successfully, I added a print statement at the end. Then I ran 
    the provided PPMI script over the tokenized data in terminal with:
        $ python ppmi.py --results_path newsPPMIresults.tsv --pairs_path 
          source.tsv --tok_path news2010.txt
    where "newsPPMIresults.tsv" is reserved for the results to be printed to, 
    "source.tsv" is a two-column .tsv of the word pairs, and "news2010.txt" is 
    the tokenized news crawl data. 
        The first run of the PPMI script printed 277 words tracked, 202 pairs 
    tracked, and 168,029,483 tokens counted. However, it showed 0 pairs were
    covered, and the --results_path returned empty.
        My second attempt at running the PPMI script was using the news crawl
    data from 2008 rather than 2010, since others in the class mentioned the [0 
    pairs covered] may have been an issue with the dataset. Unfortunately, this
    also resulted in 0 pairs covered and the --results_path returned empty.
    
>>> REVISION WRITE-UP
        With feedback from Kyle, I edited my tokenizing script to properly 
    write the tokenized data to the new file using [.join] instead of [.write].
    With this change, I was able to successfully run the PPMI.py script,
    which returned the 3-column .tsv of values. Using the script from Part 1
    and replacing the human judgements data with the PPMI results, I found the
    Path Similarity and Leacock-Chodorow methods closest to the PPMI results
    (Spearman correlation coefficient of 0.0769), and the Jiang-Conrath 
    similarity method least like the PPMI results (Spearman correlation 
    coefficient of -0.0967). Similarly to Part 1, the word pairs (eat, drink)
    and (stock, live) were not covered.

'''