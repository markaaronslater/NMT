# map a sentence to its line number in its corpus (1-based idxing), and vice versa
def create_lookup_tables(all_corpuses):
    sent_to_line = {"train.en":{}, "dev.en":{}, "train.de":{}, "dev.de":{}}
    line_to_sent = {"train.en":{}, "dev.en":{}, "train.de":{}, "dev.de":{}}

    for i, (src_sentence, trg_sentence) in enumerate(zip(all_corpuses["src"]["train.de"], all_corpuses["trg"]["train.en"])):
        sent_to_line["train.de"][src_sentence] = i+1 # 1-based indexing
        sent_to_line["train.en"][trg_sentence] = i+1 # 1-based indexing

        line_to_sent["train.de"][i+1] = src_sentence
        line_to_sent["train.en"][i+1] = trg_sentence

    for i, (src_sentence, trg_sentence) in enumerate(zip(all_corpuses["src"]["dev.de"], all_corpuses["trg"]["dev.en"])):
        sent_to_line["dev.de"][src_sentence] = i+1 # 1-based indexing
        sent_to_line["dev.en"][trg_sentence] = i+1 # 1-based indexing

        line_to_sent["dev.de"][i+1] = src_sentence
        line_to_sent["dev.en"][i+1] = trg_sentence

    return  {   "sent_to_line":sent_to_line, 
                "line_to_sent":line_to_sent
            }


#def print_lookup_tables(lookup_tables, start=1, num=5):
    


# returns lines <start> thru <start+num>, inclusive, of corpus <name>
def get_sentences(lookup_tables, name, line_nums=[1,2,3,4,5]):
    return [lookup_tables["line_to_sent"][name][line_num] for line_num in line_nums]


# returns line number, inside <name>, of each sentence in list <sentences>
def get_line_nums(lookup_tables, name, sentences):
    return [lookup_tables["sent_to_line"][name][sent] for sent in sentences]


# for each sentence in <sentences>, of corpus <name>, lookup the corresponding sentence in the opposite corpus
def get_partners(lookup_tables, name, sentences):
    # name of corresponding corpus, e.g., dev.de and dev.en are opposites
    opp_name = name[:-2] + "de" if name[-2:] == "en" else name[:-2] + "en" 
    line_nums = get_line_nums(lookup_tables, name, sentences)
    return get_sentences(lookup_tables, opp_name, line_nums)

