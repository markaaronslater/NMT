import re

########### regular expressions, lookup tables, etc. ##############
###################################################################
###################################################################

# do not convert these to lowercase even if begin a sentence or pair of quotes:
# Iwords = ["I", "I'll", "I'm", "I'd", "I've"]
# -> now handled by namesTable

# decase the words that follow end of sentence symbols:

### !!!change this to eos_symbols, or something
eos = {}
for key in [".", "!", "?", ":", "..", "...", "...."]:
    eos[key] = 1
# acList = ["Mr.", "Mrs.", "Dr.", "Ms.", "St.", "Mt.", "Lt."] # do not decase these
# -> now handled by namesTable




smartsinglequotes = re.compile(r'‘|’')
smartdoublequotes = re.compile(r'“|”')

# preceding the beginning single quote, there must be a non-word character (use assertion bc don't want it included in the match)
# following the beginning single quote, there must not be em or a digit
leftSinglequotes = re.compile(r'(?<=\W)\'(?!em|\d)(\w+)') 

# preceding the ending single quote, there must not be <in> or <s>
# following the ending single quote, there must be a non-word character (use assertion bc don't want it included in the match)
rightSinglequotes = re.compile(r'(\w+)(?<!in|.s)\'(?=\W)') 

# the naive tokenization/eos disambiguation step segments terminal periods
# from words, unless they also contain an internal period
# ex) it keeps U.S. intact, but segments Mr. into Mr .
# we stitch such acronyms back together after the fact.
# I wish I thought of a more elegant way, but it is naive, after all.
acronyms = re.compile(r'(Mr|Mrs|Dr|Ms|etc|ca|St|Mt|Lt)\s\.')
###################################################################
###################################################################
###################################################################









def tokenize_corpuses(path, corpuses):
    # corpus_names = ["train.de", "train.en", "dev.de", "dev.en", "test.de"]
    # #corpus_names = ["train.de", "train.en"]

    for corpus_name in corpuses:
        corpus = corpuses[corpus_name]
        with open(path + "tok_" + corpus_name, "w") as f:
            print("tokenizing " + corpus_name + "...")
            #tok_corpus = []
            for i, sent in enumerate(corpus):
                # replace with tokenized version
                corpus[i] = naive_tokenize(sent)
                #tok_corpus.append(sent)
                #f.write(sent + '\n') #!!! this is why references diff length than trgs, bc adding a newline
                #???maybe convert to fencepost alg, where write first sent, then '\n' + sent for each remaining sentence???

                ###!!!should do join, not string concat
                ###for now, do this compromise:
                f.write(corpus[i])
                f.write('\n')

    # tok_corpuses = []
    # for corpus_idx, corpus in enumerate(corpuses):
    #     tok_corpus = []
    #     corpus_name = corpus_names[corpus_idx]
    #     with open(path + "tok_" + corpus_name, "w") as f:
    #         print("tokenizing " + corpus_name + "...")
    #         for sent in corpus:
    #             sent = naiveTokenize(sent)
    #             tok_corpus.append(sent)
    #             f.write(sent + '\n') #!!! this is why references diff length than trgs, bc adding a newline
    #             #???maybe convert to fencepost alg, where write first sent, then '\n' + sent for each remaining sentence???

    #         tok_corpuses.append(tok_corpus)

    #return tok_corpuses





# receives str(sentence), returns tokenized str(sentence)
def naive_tokenize(sent):
    ### simplify smartquotes
    sent = re.sub(smartsinglequotes, '\'', sent)
    sent = re.sub(smartdoublequotes, '\"', sent)

    ### disambiguate single quotes from apostrophes
    sent = re.sub(leftSinglequotes, r"' \1", sent) # insert a space
    sent = re.sub(rightSinglequotes, r"\1 '", sent) # insert a space

    ### tokenize and segment sentences, keeping abbreviations and conjunctions, etc., together
    # abbr stands for abbreviation that contains internal periods. 
    #                       numbers                         abbr    ellipse      words            --         punctuation
    tokens = re.findall(r"(\.\d+|(?:\d+(?:[.,]\d+)*)(?:'?s|(?:th))?|\.\.\.*|\w+\.\w+[.\w]*|[\w']+|--|[.,:!?;\"\[\]\(\)$—-])", sent)
    sent = ' '.join(tokens)
    sent = re.sub(acronyms, r"\1.", sent) # Mr . -> Mr.

    return sent
