import sacrebleu

# estimate performance on dev set, or determine performance on test set
def evaluate(translations, references):
    bleu = sacrebleu.corpus_bleu(translations, references)
    return bleu.score