# Updated Descriptions for the CUB Dataset


Based on the descriptions created by [Scott Reed et al. Learning Deep Representations of Fine-Grained Visual Descriptions](https://github.com/reedscot/cvpr2016).  
The descriptions are updated by correcting all misspellings that cause a word to be OOV for the 6B token gloVe embeddings. Misspellings that change one valid word into another were not considered.

If you find this version of the descriptions helpful please credit the original authors:
```
@inproceedings{reed2016learning, 	
 title = {Learning Deep Representations of Fine-Grained Visual Descriptions,
 booktitle = {IEEE Computer Vision and Pattern Recognition},
 year = {2016},
 author = {Scott Reed and Zeynep Akata and Bernt Schiele and Honglak Lee},
}
```
(and give a small nod to me)

Following steps were performed:
1. Fixed encoding issues that were present in some descriptions
2. Normalize the text by expanding contractions and separating punctuation with spaces.
3. Apply substitutions (1111 unique) to replace each misspelled word with the correct one.
4. Apply some manual changes where errors were not resolveable by replacing one word with another (see ```apply_manual_processing.py```)
5. Replace "superciliary/ies" with "eyebrow(s)" as the former is not part of gloVe (if you'd rather keep it or replace even more words that are specific to bird anatomy, adapt the corresponding section in ```apply_manual_processing.py``` and run that step again)

The resulting descriptions are stored in the ```no_oov_decsriptions.json``` with the other files from that directory being the intermediate results of each fo the processing steps. All files have the structure: ```{image_id -> [list of the ten descriptions for that image id]}```

 - ```marked_for_special_processing.json``` contains information on the sentences that could not be repaired by simple substitutions.
 - ```ignored_oovs_during_fixing.json``` specifies a list of words that are not considered OOV during the process.
 - ```substitutions.json``` contains all word substitutions that were applied to the descriptions.

 To run the relabelling execute ```fix_descriptions.py``` (see its ```--help``` for more information).
 The program the queries the user for substitions for each oov word. The sentence is displayed with the offending word(s) in bold red. If there are multiple words in a sentence, the one for which the substitution is queried is underlined. Entering the appropriate substition and pressing enter adds the mapping {offending_word} -> {provided_substitution} to the set of substitutions.
 The interface takes three special inputs:
  - ```IGN``` adds the word to the list of ignored OOVs it will not be considered OOV going forwared. the list of OOVs will be written as part of the output.
  - ```BLANK``` adds the blank substitution {offending_word} -> "" to the set of substitutions.
  - the empty string (pressing Enter immediatly) adds the sentence to list of samples that are marked for later inspection
