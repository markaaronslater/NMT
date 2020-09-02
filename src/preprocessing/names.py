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
    namesTable["I"] = 1
    namesTable["I'm"] = 1
    acList = ["Mr.", "Mrs.", "Dr.", "Ms.", "St.", "Mt.", "Lt."] # do not decase these
    for ac in acList:
        namesTable[ac] = 1

    with open(path + namesFile, 'w') as f:
        for name, count in sorted(namesTable.items(), key=lambda item: item[1], reverse=True):
            f.write("{} {}\n".format(name, count))

    return namesTable, len(namesTable)

