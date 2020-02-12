CLIP_TRAILING_QUOTATION_MARKS = True
REPLACE_BIRD_WORDS = True
REPLACE_ODD_WORDS = True
APPLY_FULL_REPLACEMENTS = True
APPLY_SPECIAL_FIXES = True

# remove leading and trailing ",', ...
remove_leading_and_trailing_ids = [("141", 1), ("1398", 1), ("7911", 1),
                                   ("9180", 1), ('11218', 4), ('10963', 2)]


def remove_leading_and_trailing(sen): return sen[1:-1].strip()


# additional subsitutions. These words might be correct but are not part of gloVe - if one intends to replace them then they should also be passed as ignored words
# replace words that come up often enough to seem like they are correct words but are OOV and yield no results if searched online even in context with bird anatomy
odd_bird_words = {
    "wingular": "wing",
    "wingulars": "wings",
    "winglay": "wings",
}
# words/expressions trhat don't exist, are very specific or are formed by appending -ed to another word to express a property
# if unclear, we try to guess what the Amazon Mechanical Turk workers meant by looking at the respective picture
odd_words = {
    'whichecked': 'where it is',  # whichecked is not a word and this keeps the meaning identical
    'this sauty bird': 'this bird',
    # sauty: "adjective used to describe a situation that sucks" (Urban Dictionary) --> drop the word
    'fluffiness': 'fluff',
    'purgy': 'short',  # purgy: cute, cuddly and short (Urban Dictionary)
    'taloned feet': 'feet with talons',
    'wingbared': 'wing bars on its',
    ', long tarsused bird with': 'bird with long tarsi , ',
    'slicky blue': 'blue',  # slicky is not a word, this keeps meaning similar
    'spiculated': 'spiky',
    # spiculated is OOV but valid, substitution can be dropped if working with character based models
    'sheened': 'shimmering',  # sheened is OOV but valid, can be dropped if working with character based models
    'yellow tinging': 'a yellow tinge at'
    # tinging is OOV but valid, can be dropped if working with character based models
}
odd_words.update(odd_bird_words)
odd_word_ids = {('592', 9), ('8530', 1), ('6495', 1), ('2078', 9), ('8474', 8), ('7061', 7), ('1949', 6), ('11650', 7),
                ('3122', 4), ('1596', 9), ('1403', 2), ('3348', 9), ('339', 9), ('11429', 3), ('5919', 2), ('4907', 6),
                ('2052', 5), ('3319', 5), ('8698', 6), ('11551', 1), ('8959', 9), ('777', 7), ('9720', 4), ('247', 4),
                ('3315', 9), ('9416', 4)}

# replace bird anatomy words that are not part of the gloVe vocabulary. when using domain specific word embeddings this step should be skipped
bird_words = {
    "subciliary": "eyebrow",
    # from the context it seems as though the annotator actually meant superciliary (only occurs once)
    "subciliaries": "eyebrows",
    # from the context it seems as though the annotator actually meant superciliaries (only occurs once)
    "superciliary": "eyebrow",
    "superciliaries": "eyebrows",
    "taluses": "ankles"
    # talus refers to the ankle bone (from the encyclopedia britannica) but ankle makes more sense here
}
# these words are not OOV but it might be helpful to replace them by more commonplace words. we did NOT do this
# other_bird_words = {"tarsus","tarsi","rectrices","malar"}


# replace descriptions where we could not guess the meaning/intention of the annotator. descriptions are replaced by creating a new one.
full_replacements = {
    # image: 123.Henslow_Sparrow/Henslow_Sparrow_0046_116740.jpg - nothing about the bird is orange and "bird lactus" yields no results --> drop this description
    ("7150", 1): ("lactus", "small bird is tan and black with a small flat beak ."),
    # image: 103.Sayornis/Sayornis_0085_99503.jpg - description mentioned "red outerreachers" but nothing about the bird is red.
    ("6007", 7): ("outerreachers", "this grey bird has a white chest and has a yellow tinted belly .")
}

# fix sentences that are broken in unique ways
special_fixes = {
    # flip '"' and '.'
    ("7896",
     6): 'a small gray bird with a small , pointed yellow beak , black , pointed wings , and a dark tail shaped like a " v " .',
    # remove a random string "81043892"
    ("7142", 2): "a smaller bird with a grey and black striped belly , a short black bill , and long tail freshers .",
    # remove a comma within "colored"
    ("9638",
     3): "this is a small blue colored bird , it has a medium sized pointed beak , with a light underbelly , and a faint white wing bar .",
    # move space from to the left: "itsh ead" -> "its head"
    ("6511", 5): "this bird is white with grey on its head and has a very short beak .",
}


def apply_special_processing(img_id, i, sen):
    if CLIP_TRAILING_QUOTATION_MARKS and (img_id, i) in remove_leading_and_trailing_ids:
        return remove_leading_and_trailing(sen)
    if REPLACE_ODD_WORDS and (img_id, i) in odd_word_ids:
        return replace_words(sen, odd_words)
    if APPLY_FULL_REPLACEMENTS and (img_id, i) in full_replacements:
        return full_replacements[(img_id, i)][1]
    if APPLY_SPECIAL_FIXES and (img_id, i) in special_fixes:
        return special_fixes[(img_id, i)]
    return sen


def replace_words(sen, replace_dict):
    for w in replace_dict:
        if w in sen:
            sen = sen.replace(w, replace_dict[w])
    return sen


def apply_special_processing_whole_dataset(data):
    updated = dict()
    for img_id in data:
        updated[img_id] = list()
        for i, sen in enumerate(data[img_id]):
            s = apply_special_processing(img_id, i, sen)
            if REPLACE_BIRD_WORDS:
                s = replace_words(s, bird_words)
            updated[img_id].append(s)
    return updated
