import torch

from src.preprocessing.truecase import truecase_sentence, should_lower_de
from src.predict import predict
from src.preprocessing.build_batches import get_test_batches
from src.preprocessing.apply_vocab import to_indices


# translate list of German test sentences in end-to-end fashion.
# -> performs entire preprocessing, prediction, and postprocessing
# pipelines on a set of user-inputted German sentences (for demo), or an
# entire test set (for replicating BLEU results).

# input is list of German sentences as strings 
def translate(input, stanza_processor, translator, src_word_to_idx, idx_to_trg_word,
            bpe, device='cuda:0', bsz=32):
    print("tokenizing, multiword-token-expanding, pos-tagging...")
    doc = stanza_processor('\n\n'.join(input)) # returns Document object
    sentences = doc.sentences # list of Stanza Sentence objects
    truecased_sentences = []
    print("truecasing...")
    for sent in sentences:
        truecase_sentence(sent, should_lower_de)
        truecased_sentences.append(' '.join([word.text for word in sent.words]))

    print("subword segmenting...")
    subword_segmented_sentences = [bpe.process_line(sent).split() for sent in truecased_sentences]
    print("converting to batches of indices...")
    to_indices("user-input", subword_segmented_sentences, src_word_to_idx) # modifies in-place
    test_batches = get_test_batches(subword_segmented_sentences, bsz, device)
    print("making predictions...")
    translations, _, _ = predict(translator, test_batches, idx_to_trg_word)

    return translations