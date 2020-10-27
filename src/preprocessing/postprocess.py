# translation_idx_pairs is list of pairs, where:
# 1st component is (bsz x T) tensor, where T is number of decode time steps (at least 1, and at most max_src_len + decode_slack), holding the predicted translations.
# 2nd component is (bsz,) tensor, where entry i holds line of source corpus that i'th translation of component 1 is for.
def post_process(translation_batches, idx_to_trg_word, eos_idx):
    translation_idx_pairs = [] # list of pairs where 1st component is list-form of translation, and 2nd component is source corpus line number of sentence it is a translation for
    for (translation, corpus_indices) in translation_batches:
        translation = extract_translation(translation.tolist(), eos_idx)
        translation_idx_pairs += list(zip(translation, corpus_indices.to_list()))

    # unsort them so line up with dev trg sentences during evaluation.
    translation_idx_pairs = sorted(translation_idx_pairs, key = lambda pair: pair[1])

    # convert to words. no longer need the corpus indices.
    # ("idx" in idx_to_trg_word refers to index in target vocabulary mapping)
    translations = [[idx_to_trg_word[i] for i in translation] for translation, _ in translation_idx_pairs]

    # remove concatenated pos-tags, if applicable


    # naively reapply casing based on sentence and double-quote segmentation, and then detokenize (convert each sentence from a list of words into a string).
    # -> sacrebleu will "fully detokenize" them for you, e.g., convert I' ve -> I've
    # ??but will it do 20 - year - old -> 20-year-old??
    # (see formatPrediction)
    translations = [' '.join(naive_recase(translation)) for translation in translations]

    # desegment subwords back into words, if applicable

    return translations


# translation is list of lists of decoder predictions, each of same length. for each sentence, find location of predicted eos symbol, and remove every predicted token that follows it (i.e., "extract" the translation from entire decoder output). end up with list of variable-length translations. 
def extract_translation(translation, eos_idx):
    extracted_translations = []
    for j in range(len(translation)):
        try:
            eos_position = translation[j].index(eos_idx) # keep translation up to but not including eos
        except ValueError:
            eos_position = len(translation[j]) # never produced eos (decoder "timed out"), so keep entire translation
        
        extracted_translations.append(translation[j][:eos_position])

    return extracted_translations


# when passed a sentence (which can actually consist of multiple linguistic sentences) represented as a list of words, capitalizes:
# -first word of every sentence, 
# -first word following every opening double quote. simplifying assumptions:
#       -for any sent with odd number of quotes, the unpaired quote comes at the end, not the beginning (will be correct ~50% of the time, presumably),
#       -the first word of the quote ought to be capitalized (usually, but not always the case)
# -first word following every closing double quote, if last word of double quote was an end-of-sentence symbol.

# ex) " that's right , " he said . and then I left .
# ->  " That's right , " he said . And then I left .

# ex) I said , " yes , sir . I did . " and we started arguing .
# ->  I said , " Yes , sir . I did . " And we started arguing .
eos_symbols = {".", "!", "?", ":", "..", "...", "...."}
def naive_recase(sent):
    if not sent:
        return sent # if predicted 'empty-sentence' (immediately predicted eos token)

    # all locations of sentence containing an eos symbol, other than the final one (which has no word following it, so nothing to capitalize)
    eos_positions = [i for i, word in enumerate(sent) if word in eos_symbols and i != len(sent)-1]
    quote_positions = [i for i, word in enumerate(sent) if word == '"' and i != len(sent)-1]

    # capitalize first word of each sentence
    sent[0] = sent[0].capitalize()
    for j in eos_positions:
        sent[j+1] = sent[j+1].capitalize()

    # capitalize the first word inside each pair of double-quotes
    for i in quote_positions[::2]: 
        sent[i+1] = sent[i+1].capitalize()
    
    # capitalize the first word after each pair of double quotes (only if
    # word before endquote was an eos symbol)
    for j in quote_positions[1::2]: 
        if j-1 in eos_positions:
            sent[j+1] = sent[j+1].capitalize()

    return sent