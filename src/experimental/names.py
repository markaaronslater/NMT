from collections import Counter

def createNamesTable(path, namesFile, train_sentences): # list of str(sentence)'s
    namesTable = Counter() # names/proper nouns map to their frequency in corpus, everything else maps to 0
    lower_train_sentences = []
    print("creating " + namesFile[-2:] + " names table...")
    # get vocab counter for num capitalized occurrences of words
    capWords = Counter()
    for sent in train_sentences:
        lower_train_sent = []
        tokens = sent.split()
        for word in tokens:
            if word[0].isupper():
                capWords[word] += 1
                #print(capWords)
            lower_train_sent.append(word.lower())
        lower_train_sentences.append(lower_train_sent)

    #print(lower_train_sentences)
    allWords = Counter()
    # get vocab counter for total occurrences of words 
    # (both capitalized and lowercase)
    for sent in lower_train_sentences:
        for word in sent:
            allWords[word] += 1

    for word in capWords:
        #print(word)
        if allWords[word.lower()] == capWords[word]:
            # num times ever occurred == num times occurred in capitalized form,
            # so assume it is a name or proper noun 
            namesTable[word] = capWords[word]
        
    # want to track all words that should not be decased in namesTable,
    # so include certain acronyms like Mr.,
    # and fix bug where corpus had a few occurrences of i and i'm

    ### ???why am i not also including I've, I'd, I'll, etc., here??
    ### i think bc they occur in lower case a few times.
    namesTable["I"] = 1
    namesTable["I'm"] = 1
    acList = ["Mr.", "Mrs.", "Dr.", "Ms.", "St.", "Mt.", "Lt."] # do not decase these
    for ac in acList:
        namesTable[ac] = 1

    with open(path + namesFile, 'w') as f:
        for name, count in sorted(namesTable.items(), key=lambda item: item[1], reverse=True):
            f.write("{} {}\n".format(name, count))

    return namesTable, len(namesTable)





def create_names_tables(path, corpuses): # list of str(sentence)'s
    lower_occurrences = Counter() # num times each word occurred in lower case form. if > 0, then should include a lower case version in vocab.
    #syn_cap_occurrences = Counter() # num times each word occurred in capitalized form, due to a syntactic reason (e.g., bc occurred at beginning of sentence, etc.)
    sem_cap_occurrences = Counter() # num times each word occurred in capitalized form, due to semantic reason (i.e., not for a syntactic reason). if > 0, then should include a capitalized version in vocab, bc can occur as a name / proper noun.
    #total_occurrences = Counter() 
    # src sentences
    print("creating de names table...")


    if lower_occurrences > 0:
        # occurred as a non-name/proper noun at some point, so include lower case version in vocab

    if sem_cap_occurrences > 0:
        # occurred as a name/proper noun at some point, so include capitalized version in vocab
        # (does not handle the rare edge case where a word that isn't a name/proper noun happens to always occur, e.g., at the start of the sentence, causing it to always be treated as if capitalized only due to syntax)


    # heuristic: in the second pass, when encounter a word at a location where should be capitalized bc of syntax, if CAN be capitalized (the word belongs to sem_cap_occurrences), i will assume that it SHOULD be capitalized, and therefore not decase it

    # at least first letter is upper-case, i.e., word is capitalized, or follows ", :, or an eos symbol, such as ., !, or ? (i.e., the "sentence" is actually multiple sentences)

    for i, sent in enumerate(corpuses["train.de"]): 
        # skip first word of each sent, which always capitalized due to syntax, rather than bc a name/proper noun.
        for j, word in enumerate(sent)[1:]:
            if word[0].isupper() and sent[j-1] not in eos:
                
                sem_cap_occurrences[word] += 1
            #else:
            #    lower_occurrences[word] += 1

    # add all lower case words to 
    # make 2nd pass thru train.de, decasing all words, even those that ought to be capitalized due to syntax (e.g., first words of sentence), unless its capitalized version exists in the vocab (i.e., the word belongs to sem_cap_occurrences)






    lower_train_sentences = []
    # get vocab counter for num capitalized occurrences of words
    capWords = Counter()
    for sent in train_sentences:
        lower_train_sent = []
        tokens = sent.split()
        for word in tokens:
            if word[0].isupper():
                capWords[word] += 1
                #print(capWords)
            lower_train_sent.append(word.lower())
        lower_train_sentences.append(lower_train_sent)

    #print(lower_train_sentences)
    allWords = Counter()
    # get vocab counter for total occurrences of words 
    # (both capitalized and lowercase)
    for sent in lower_train_sentences:
        for word in sent:
            allWords[word] += 1



    namesTable = Counter() # names/proper nouns map to their frequency in corpus, everything else maps to 0

    for word in capWords:
        #print(word)
        if allWords[word.lower()] == capWords[word]:
            # num times ever occurred == num times occurred in capitalized form,
            # so assume it is a name or proper noun 
            namesTable[word] = capWords[word]
        
    # want to track all words that should not be decased in namesTable,
    # so include certain acronyms like Mr.,
    # and fix bug where corpus had a few occurrences of i and i'm

    ### ???why am i not also including I've, I'd, I'll, etc., here??
    ### i think bc they occur in lower case a few times.
    namesTable["I"] = 1
    namesTable["I'm"] = 1
    acList = ["Mr.", "Mrs.", "Dr.", "Ms.", "St.", "Mt.", "Lt."] # do not decase these
    for ac in acList:
        namesTable[ac] = 1

    with open(path + namesFile, 'w') as f:
        for name, count in sorted(namesTable.items(), key=lambda item: item[1], reverse=True):
            f.write("{} {}\n".format(name, count))

    return namesTable, len(namesTable)

