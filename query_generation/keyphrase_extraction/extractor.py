"""Contain method that return list of candidate"""

import re
import numpy as np

SALIENT_NER_TYPES = {
    'PERSON',
    'NORP',
    'FAC',
    'ORG',
    'GPE',
    'LOC',
    'PRODUCT',
    'EVENT',
    'WORK_OF_ART',
    'LAW',
    'LANGUAGE'
}


def get_keyphrase_idf_score(keyphrase, idf_model):
    max_idf = max(idf_model.idf_)
    min_idf = min(idf_model.idf_)
    scale_idf = max_idf - min_idf
    scores = list()
    for word in keyphrase:
        word = word.lower()
        if word in idf_model.vocabulary_:
            idf_ = idf_model.idf_[idf_model.vocabulary_[word]]
            score = (idf_ - min_idf) / scale_idf
            scores.append(score)
    if len(scores) > 0:
        return np.mean(scores)
    return 1.0


def extract_candidates(raw_text: str, ptagger, idf_model, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param raw_text: A string containing the raw text to extract
    :param ptagger: A Pos Tagger object see @PosTagger
    :param idf_model: A TfidfVectorizer object see @PosTagger
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return: list of candidate phrases (string)
    """

    keyphrase_candidates = list()
    keyphrase_candidate_idfs = list()
    doc = ptagger.nlp(raw_text.replace(r'`', r"'"))
    candidate_chunks = list(doc.noun_chunks) + list(doc.ents)
    for chunk in candidate_chunks:
        words_in_chunk = chunk.text.split()
        start_idx = 0
        for idx, word in enumerate(words_in_chunk):
            if word in ptagger.stop_words:
                start_idx += 1
            else:
                break
        chunk_doc = ptagger.nlp(' '.join(words_in_chunk[start_idx:]))
        keyphrase = list()
        phrase_type = 0  # 0 refers to non-PROPN 1 refers to PROPN
        entity = False
        for token in chunk_doc:
            if token.text == '-' or token.tag_ == 'HYPH':
                keyphrase.append(token.text)
            elif token.text == "'s" or token.text == "'" or token.pos_ == 'PUNCT':
                if keyphrase is not None and ' '.join(keyphrase) not in keyphrase_candidates:
                    keyphrase_candidates.append(' '.join([k for k in keyphrase if k != "'s"]))
                    keyphrase_candidate_idfs.append(get_keyphrase_idf_score(keyphrase, idf_model))
                keyphrase = list()
            elif token.tag_ == 'NNP' or token.tag_ == 'NNPS':
                if phrase_type == 0:
                    if keyphrase is not None and ' '.join(keyphrase) not in keyphrase_candidates:
                        keyphrase_candidates.append(' '.join(keyphrase))
                        keyphrase_candidate_idfs.append(get_keyphrase_idf_score(keyphrase, idf_model))
                    if token.ent_iob_ == 'B' and token.ent_type_ in SALIENT_NER_TYPES:
                        entity = True
                    keyphrase = [token.text]
                    phrase_type = 1
                else:
                    if token.ent_iob_ == 'B' and token.ent_type_ in SALIENT_NER_TYPES:
                        if keyphrase is not None and ' '.join(keyphrase) not in keyphrase_candidates:
                            keyphrase_candidates.append(' '.join(keyphrase))
                            keyphrase_candidate_idfs.append(get_keyphrase_idf_score(keyphrase, idf_model))
                        entity = True
                        keyphrase = list()
                    elif entity and token.ent_iob_ == 'O':
                        if keyphrase is not None and ' '.join(keyphrase) not in keyphrase_candidates:
                            keyphrase_candidates.append(' '.join(keyphrase))
                            keyphrase_candidate_idfs.append(get_keyphrase_idf_score(keyphrase, idf_model))
                        entity = False
                        keyphrase = list()
                    keyphrase.append(token.text)
            else:
                if phrase_type == 0:
                    keyphrase.append(token.text)
                else:
                    if keyphrase is not None and ' '.join(keyphrase) not in keyphrase_candidates:
                        keyphrase_candidates.append(' '.join(keyphrase))
                        keyphrase_candidate_idfs.append(get_keyphrase_idf_score(keyphrase, idf_model))
                    keyphrase = [token.text]
                    phrase_type = 0
        if keyphrase is not None and ' '.join(keyphrase) not in keyphrase_candidates:
            keyphrase_candidates.append(' '.join(keyphrase))
            keyphrase_candidate_idfs.append(get_keyphrase_idf_score(keyphrase, idf_model))

    keyphrase_candidate_idfs = [ner for ner, kp in zip(keyphrase_candidate_idfs, keyphrase_candidates) if len(kp.split()) <= 5]
    keyphrase_candidates = [kp for kp in keyphrase_candidates if len(kp.split()) <= 5]

    return keyphrase_candidates, keyphrase_candidate_idfs


def extract_sent_candidates(text_obj):
    """
    :param text_obj: input Text Representation see @InputTextObj
    :return: list of tokenized sentence (string) , each token is separated by a space in the string
    """
    return [(' '.join(word for word, tag in sent)) for sent in text_obj.pos_tagged]


def unique_ngram_candidates(strings):
    """
    ['machine learning', 'machine', 'backward induction', 'induction', 'start'] ->
    ['backward induction', 'start', 'machine learning']
    :param strings: List of string
    :return: List of string where no string is fully contained inside another string
    """
    results = []
    for s in sorted(set(strings), key=len, reverse=True):
        if not any(re.search(r'\b{}\b'.format(re.escape(s)), r) for r in results):
            results.append(s)
    return results
