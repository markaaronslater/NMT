

def decaseCorpuses(corpuses, path, de_namesTable, en_namesTable):
    corpus_names = ["train.de", "train.en", "dev.de", "dev.en", "test.de"]
    #corpus_names = ["train.de", "train.en"]

    decased_corpuses = []

    for corpus_idx, corpus in enumerate(corpuses):
        decased_corpus = []
        corpus_name = corpus_names[corpus_idx]
        namesDict = de_namesTable if corpus_name[-2:] == "de" else en_namesTable
    
        with open(path + "decased_" + corpus_name, "w") as f:
            print("decasing " + corpus_name + "...")
            for sent in corpus:
                sent = naiveDecase(sent, namesDict)
                decased_corpus.append(sent)
                f.write(sent + '\n')
            decased_corpuses.append(decased_corpus)
    
    return decased_corpuses






def naiveDecase(sent, namesDict):
    if not sent:
        return sent # for debugging empty sentence

    sent = sent.split()
    #if sent[0] not in Iwords and sent[0] not in acList:

    # lower case first word of the sentence:
    if sent[0] not in namesDict:
        sent[0] = sent[0][0].lower() + sent[0][1:] # this handles leading acronyms, like people's names, followed by :
    # ex) BG -> bG, not bg, so that when predict this during inference, can recase to BG, not Bg
    ### (this format is common in the corpus, bc transcripts of TED-talks, where diff speakers will precede given sentences, in a dialogue exchange, etc.)

    positions = [i for i,word in enumerate(sent) if (word in eos or word == '"') and i != len(sent)-1]
    #quotePositions = [i for i,word in enumerate(sent) if word == '"' and i != len(sent)-1]
    # (do not extract double quote at last position of sentence, bc nothing follows it)
    
    # in decasing, no need to discriminate between quote openers and closers
    # lowercase the first word inside a pair of doublequotes,
    # and first word to follow a pair of doublequotes or a lone period, exclamation point, or question mark
    for j in positions: 
        if sent[j+1] not in namesDict:
            sent[j+1] = sent[j+1].lower()
    # makes simplifying assumption that for any sent with odd number of quotes,
    # the unpaired quote comes at the end, not the beginning
    # will be correct ~50% of the time, presumably.



    return ' '.join(sent)














# wrapper method for adding casing back to produced targets, and
# addressing fact that moses detokenizer does not correctly handle -
# given list of str(words), produces str(sentence)
def formatPrediction(sent):
    if not sent: # its an empty list, so return an empty string
        return '' # for debug, empty sentences

    sent = naiveRecase(sent) # list of str(words)
    sent = ' '.join(sent) # str(sentence)
    if "-" in sent:
        sent = sent.replace(" - ", "-")

    return sent





