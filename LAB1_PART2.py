### LING83600: LANGUAGE TECHNOLOGY - LAB 1 PART 2
### by Winnie Yan

### Write-up at bottom 

import nltk

# Assign variable to open news data
data = "news.2008.en.shuffled.deduped"
data = open(data, "r", encoding = "utf-8")

# Tokenize using NLTK word tokenizer and write to new .txt file
with open("news2008.txt", "w", encoding = "utf-8") as f:
    for line in data:
        f.write("%s\n" % (nltk.word_tokenize(line)))
    print("Complete")

# After tokenization complete, run provided PPMI script over tokenized data.

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

'''
>>> Spearman Correlation Coefficient (to 4 digits):

    
>>> WORD PAIRS NOT COVERED (ERRORS AND RESOLUTIONS)
    

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
'''