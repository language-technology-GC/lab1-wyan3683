### LING83600: LANGUAGE TECHNOLOGY - LAB 1 PART 3
### by Winnie Yan

### Write-up at bottom 



# Install Gensim 
# Run provided word2vec.py script over tokenized data from part 2
# $ python word2vec.py --results_path newsPPMIresults.tsv --pairs_path 
#   source.tsv --tok_path news2010.txt

# INFO: training on a 840147415 raw words (641288431 effective words) took 
#   2767.8s, 231700 effective words/s
# word2vec.py:28: DeprecationWarning: Call to deprecated `similarity` (Method 
#   will be removed in 4.0.0, use self.wv.similarity() instead).
# score = round(w2v.similarity(x, y), 6)


'''
>>> Spearman Correlation Coefficient (to 4 digits):

    
>>> WORD PAIRS NOT COVERED (ERRORS AND RESOLUTIONS)
    

>>> WRITE-UP (DESCRIPTION OF APPROACH & PROBLEMS ENCOUNTERED)
    
        The first run of the word2vec.py script using the same .tsv files from
    part 2's first run of the ppmi.py script did not yield useable results.
        
'''