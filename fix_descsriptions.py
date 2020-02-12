import json
import re
from typing import List, Dict, Set, Tuple
import datetime
import sys
import apply_manualprocessing

r = re.compile(' {2,}')

replacements = dict()
ignored = set()
marked = []
marked_sens = set()


class bcolors:
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main(vocab, dataset, ignored_file=None, replacements_file=None, marked_file=None, out_folder='/tmp'):
    global replacements
    global ignored
    global marked
    global marked_sens
    start_stamp = datetime.datetime.utcnow().isoformat().replace(':', '-')[:-7]
    try:
        vocab: Set[str] = set(json.load(open(vocab)))
        data: Dict[int, List[str]] = json.load(open('../datasets/imgid_2_descriptions.json'))
        if replacements_file is not None:
            print('Loaded substitutions file from:', replacements_file)
            replacements.update(json.load(open(replacements_file)))
        if ignored_file is not None:
            ignored = set(json.load(open(ignored_file)))
            print('Start with ignored words: ', ignored)
        if marked_file is not None:
            marked = json.load(open(marked_file))
            marked_sens = {(entry[0], entry[1]) for entry in marked}
            print(f'Start with {len(marked)} sentences marked for special processing')

        normalized = normalize_descriptions(data)

        oov_ids = find_oov(normalized, vocab)
        oov_words = [w for l in map(lambda tup: tup[-1], oov_ids) for w in l]

        print(f'Dataset has {len(oov_words)} words ({len(set(oov_words))} unique) not in the glove vocabulary.')
        print(f'Dataset has {len(oov_ids)} descriptions with oovs.')
        input(
            'Start fixing? Enter word to replace underlined word. "IGN" to ignore the OOV error, "MRK" to mark sentence for later processing.')
        updated = fix_descriptions(normalized, vocab)
        updated_processed = apply_manualprocessing.apply_special_processing_whole_dataset(updated)

        updated_words = set(' '.join([sen for l in updated_processed.values() for sen in l]).split())
        if len(updated_words - vocab) == 0:
            print('No more OOV!')
        else:
            print(f'{len(updated_words - vocab)} OOV left: {updated_words - vocab}')

    finally:
        json.dump(normalized, open(f'{out_folder}/normalized_{start_stamp}.json', 'w'))
        json.dump(oov_ids, open(f'{out_folder}/oov_descriptions_{start_stamp}.json', 'w'), indent=2)
        json.dump(replacements, open(f'{out_folder}/substitutions_{start_stamp}.json', 'w'))
        json.dump(list(ignored), open(f'{out_folder}/ignored_oovs_{start_stamp}.json', 'w'))
        json.dump(marked, open(f'{out_folder}/marked_for_processing_{start_stamp}.json', 'w'), indent=2)
        json.dump(updated, open(f'{out_folder}/updated_descriptions_{start_stamp}.json', 'w'), indent=2)
        json.dump(updated_processed, open(f'{out_folder}/clean_descriptions{start_stamp}.json', 'w'), indent=2)


def fix_descriptions(data, vocab):
    updated = dict()
    for c, img_id in enumerate(data):
        updated[img_id] = list()
        for i, sen in enumerate(data[img_id]):
            new_sen = fix_description(sen, vocab, f'{c:5}.{i}/{len(data) - 1}.9 - img:{img_id}', img_id, i)
            updated[img_id].append(new_sen)
    return updated


def fix_description(sen, vocab, top_str, img_id, i):
    global marked
    global replacements
    global ignored
    global marked_sens

    words = set(sen.split())
    print('\033[2J\033[H', end='')
    print(top_str)
    if len(words - vocab - ignored) == 0:
        return sen

    else:
        error_words = words - vocab
        while any([w in replacements for w in set(sen.split())]):
            for w in [w for w in set(sen.split()) if w in replacements]:
                sen = sen.replace(w, replacements[w])
        if (img_id, i) in marked_sens:
            return sen
        words = set(sen.split())
        exempt = set()
        error_words = words - vocab
        while len(error_words - ignored - exempt):
            err_w = next(iter(error_words))
            print(f'{bcolors.ENDC}\033[2J\033[H', end='')
            print(top_str)
            print('Ignored words: ', ignored)
            print(' '.join([w if w not in (
                error_words) else f'{bcolors.BOLD}{bcolors.FAIL if w not in ignored else ""}{bcolors.UNDERLINE if w == err_w else ""}{w}{bcolors.ENDC}'
                            for w in sen.split()]))
            repl_word = input(bcolors.UNDERLINE)
            if repl_word == 'IGN':
                ignored.add(err_w)
                continue
            elif repl_word == '':
                exempt.add(err_w)
                marked.append((img_id, i, err_w, sen))
                marked_sens.add(sen)
                return sen
            elif repl_word == 'BLANK':
                repl_word = ''

            replacements[err_w] = repl_word
            while any([w in replacements for w in set(sen.split())]):
                for w in [w for w in set(sen.split()) if w in replacements]:
                    sen = sen.replace(w, replacements[w])
            words = set(sen.split())
            error_words = words - vocab
    return sen


def gather_oov_statistic(oovs):
    counts = dict()
    for w in oovs:
        counts[w] = counts.get(w, 0) + 1
    return counts


def normalize_descriptions(data):
    updated = dict()
    for img_id in data:
        updated[img_id] = list()
        for descr in data[img_id]:
            # normalize text and fix encoding errors
            descr = descr.replace('be;;y', 'belly')  # would be plit by punctuation normalization
            descr = descr.replace(b'\xef\xbf\xbd'.decode(), ' ').replace('\x1b[6~', '')  # fix encoding errors
            # expand contractions 
            for cntrct in filter(lambda w: w in descr, contraction_dict):
                descr = descr.replace(cntrct, contraction_dict[cntrct])
            # normalize punctionation (except "'s")
            descr = descr.replace("'s", "<APS>")
            descr = descr.replace('.', ' . ').replace(',', ' , ').replace('-', ' - ').replace('/', ' / ')
            descr = descr.replace(')', ' ) ').replace('(', ' ( ').replace(
                ';', ' ; ').replace(':', ' : ').replace("'", " ' ").replace('!', ' ! ').replace('?', ' ? ')
            descr = descr.replace("<APS>", " 's").strip()
            # normalize space
            descr = r.sub(' ', descr)
            updated[img_id].append(descr)
    return updated


def find_oov(data, vocab):
    oovs = list()
    for img_id in data:
        for i, descr in enumerate(data[img_id]):
            words = set(descr.split())
            oov = words - vocab
            if len(oov) > 0:
                oovs.append((img_id, i, list(oov)))
    return oovs


contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                    "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                    "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                    "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
                    "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                    "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                    "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                    "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                    "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                    "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
                    "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
                    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--ignored', help='json file with a list of oov that should not be replaced')
    parser.add_argument('--marked',
                        help='json file with sentences that have to be processed by hand (as generated by this script)')
    parser.add_argument('--substitutions', help='pre-initialize subsitution dictionary from this json file ( {"from" : "to", ...})')
    parser.add_argument('--vocab', default='../datasets/vocab_words.json', help='json list of vocabulary')
    parser.add_argument('--dataset', default='../datasets/imgid_2_descriptions.json',
                        help='dataset of structure { img_id_1 : [description_1,...,description_n], ... }')
    parser.add_argument('--out', default='/tmp', help='file destionation')
    args = parser.parse_args()

    main(vocab=args.vocab, dataset=args.dataset, ignored_file=args.ignored, replacements_file=args.substitutions,
         marked_file=args.marked, out_folder=args.out)
