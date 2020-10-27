# -remove songs
# -remove any pair where at least one of the sentences is over length 100
def filter_corpuses(corpuses):
    filtered_src_sentences = []
    filtered_trg_sentences = []
    num_overly_long = 0 # number of sentences that were over 100 but not songs
    num_songs = 0
    print(f'{len(corpuses["train.de"])} <src sent, trg sent> pairs before filtering')

    for i in range(len(corpuses["train.de"])):
        src_sent = corpuses["train.de"][i]
        trg_sent = corpuses["train.en"][i]

        if '♫' in src_sent or '♫' in trg_sent or '♪' in src_sent or '♪' in trg_sent:
            num_songs += 1
        elif len(src_sent.split()) > 100 or len(trg_sent.split()) > 100:
            num_overly_long += 1
        else:
            filtered_src_sentences.append(src_sent)
            filtered_trg_sentences.append(trg_sent)

    # overwrite original train corpuses
    corpuses["train.de"] = filtered_src_sentences
    corpuses["train.en"] = filtered_trg_sentences

    print(f"{num_songs} songs, {num_overly_long} overly long")
    print(f'{len(corpuses["train.de"])} <src sent, trg sent> pairs after filtering')