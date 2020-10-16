### LING83600: LANGUAGE TECHNOLOGY - LAB 1 PART 3
### by Winnie Yan

### Write-up at bottom 

# Install Gensim 
# Run provided word2vec.py script over tokenized data from part 2
# $ python word2vec.py --results_path newsW2Vresults.tsv --pairs_path 
#   source.tsv --tok_path news2010.txt

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
    data = pandas.read_csv("newsW2Vresults.tsv", delimiter = "\t")
        
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

# INFO: training on a 840095890 raw words (627002655 effective words) took 
# 2433.2 s, 257683 effective words/s


'''
>>> Spearman Correlation Coefficient (to 4 digits):
    INFO: path_sim	0.3623
    INFO: lea_chod	0.3623
    INFO: wu_palm	0.4246
    INFO: res	0.5132
    INFO: ji_con	0.3457
    INFO: lin	0.4451
    
>>> WORD PAIRS NOT COVERED (ERRORS AND RESOLUTIONS)
    (eat, drink)
    (stock, live)

>>> WRITE-UP (DESCRIPTION OF APPROACH & PROBLEMS ENCOUNTERED)
    
        The first run of the word2vec.py script using the same .tsv files from
    part 2's first run of the ppmi.py script did not yield useable results.
        After making changes in Part 2 and yielding proper results for the
    PPMI.py script, I ran part 3 again, resulting in valid output. Calculting
    the Spearman coeffficient correlations with the Word2Vec results replacing
    the human judgments data indicated that the Resnick similarity method was 
    closest to the Word2Vec results (at 0.5132) while the Path Similarity and 
    Leacock-Chodorow similarity methods were least similar to the Word2Vec 
    results (at 0.3623). Once again, the word pairs (eat, drink) and (stock, 
    live) had to be removed for the script to run efficiently.
'''