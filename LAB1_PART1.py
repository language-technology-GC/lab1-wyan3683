### LING83600: LANGUAGE TECHNOLOGY - LAB 1 PART 1
### by Winnie Yan 

### Write-up at bottom

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
    data = pandas.read_csv("sauce.tsv", delimiter = "\t")
        
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

''' 
>>> Spearman Correlation Coefficient (to 4 digits):
    
    INFO: path_sim	0.4537
    INFO: lea_chod	0.4537
    INFO: wu_palm	0.5457
    INFO: res	0.5827
    INFO: ji_con	0.4841
    INFO: lin	0.5693


>>> WORD PAIRS NOT COVERED (ERRORS AND RESOLUTIONS)
    
    All word pairs were covered except (drink, eat) and (stock, live).
    
    WordNetError: Computing the lch similarity requires Synset('drink.n.01') 
        and Synset('eat.v.01') to have the same part of speech.
    Manually remove (drink, eat) pair from 'sauce.tsv'.

    WordNetError: Computing the lch similarity requires Synset('stock.n.01') 
        and Synset('populate.v.01') to have the same part of speech.
    No (stock, populate) pair in 'sauce.tsv'. 
    Try removing individual (stock, __) pairs to find the pair causing error. 
    Error caused by (stock, live); pair manually removed from 'sauce.tsv'.
    

>>> WRITE-UP (DESCRIPTION OF APPROACH & PROBLEMS ENCOUNTERED)
        
        For Part 1 of the lab, I was unsure how to start so I looked for 
    similar problems online, and I looked at the ppmi.py script provided for 
    Part 2. I decided to use a similar approach, adapting the use of the 
    logging.info. I also created a class to raise errors, and a class to 
    compute the similarity of word pairs to be used in the main function. 
    Calculating the Spearman correlation coefficient was added as a separate 
    function before creating the main function. To casefold, I added column 
    names manually and then casefolded the first two columns (column 3 did not
    require casefolding). Then I made and appended the list of synset pairs, 
    choosing the first lemma, as chosen in the Calculate_Similarity class. 
    After adding the similarity scores, I used logging.info to calculate 
    correlations of the similarity scores. 
        Some errors were raised when running of the code (mentioned above).
    The pairs (drink, eat) and (stock, populate) raised errors for having 
    different parts of speech tags in all methods except the Path Similarity
    method and the Wu-Palmer method (found by inserting [try/except print...] 
    statements at each method within the class). Manually removing the 
    (drink, eat) pair prevented the error from being raised again. The 
    (stock, populate) pair could not be found in the dataset so I tried 
    removing the (stock, __) pairs one at a time until the error stopped 
    showing up, which revealed (stock, live) as the offending pair.
        Results showed the Resnik similarity method having the highest 
    correlation coefficient (0.5827), and the path similarity and Leacock-
    Chodorow methods having the lowest correlation coefficient (at 0.4537),
    meaning the Resnik similarity method was closest to the human judgements
    of similarity, and the path similarity and Leacock-Chodorow methods were 
    least similar to the human judgements of similarity.
        An unexpected problem encountered was breaking my laptop and having to
    borrow someone's to complete the assignment. As a result, I spent a lot of 
    time installing components needed to complete the lab and redoing all the 
    parts because none of the parts I had completed were saved to a cloud. For 
    future precautions, I will be more diligent in saving completed work online
    to prevent this from happening again.
    
'''
