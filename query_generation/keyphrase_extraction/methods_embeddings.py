import numpy as np

from query_generation.keyphrase_extraction.extractor import extract_candidates, extract_sent_candidates


def extract_doc_embedding(embedding_distrib, inp_rpr, use_filtered=False):
    """
    Return the embedding of the full document
    :param embedding_distrib: embedding distributor see @EmbeddingDistributor
    :param inp_rpr: input text representation see @InputTextObj
    :param use_filtered: if true keep only candidate words in the raw text before computing the embedding
    :return: numpy array of shape (1, dimension of embeddings) that contains the document embedding
    """
    if use_filtered:
        tagged = inp_rpr.filtered_pos_tagged
    else:
        tagged = inp_rpr.pos_tagged

    tokenized_doc_text = ' '.join(token[0].lower() for sent in tagged for token in sent)
    return embedding_distrib.get_tokenized_sents_embeddings([tokenized_doc_text])


def extract_candidates_embedding_for_doc(embedding_distrib, raw_text, ptagger, idf_model):
    """
    Return the list of candidate phrases as well as the associated numpy array that contains their embeddings.
    Note that candidates phrases extracted by PosTag rules  which are uknown (in term of embeddings)
    will be removed from the candidates.
    :param embedding_distrib: embedding distributor see @EmbeddingDistributor
    :param raw_text: A string containing the raw text to extract
    :param ptagger: A Pos Tagger object see @PosTagger
    :param idf_model: A TfidfVectorizer object see @PosTagger
    :return: A tuple of two element containing 1) the list of candidate phrases
    2) a numpy array of shape (number of candidate phrases, dimension of embeddings :
    each row is the embedding of one candidate phrase
    """
    candidates, idf_scores = np.array(extract_candidates(raw_text, ptagger, idf_model))
    if len(candidates) > 0:
        candidates_lower = [c.lower() for c in candidates]
        embeddings = np.array(embedding_distrib.get_tokenized_sents_embeddings(candidates_lower))  # Associated embeddings
        valid_candidates_mask = ~np.all(embeddings == 0, axis=1)  # Only candidates which are not unknown.
        return candidates[valid_candidates_mask], embeddings[valid_candidates_mask, :], idf_scores[
            valid_candidates_mask]
    else:
        return np.array([]), np.array([]), np.array([])


def extract_sent_candidates_embedding_for_doc(embedding_distrib, inp_rpr):
    """
    Return the list of candidate senetences as well as the associated numpy array that contains their embeddings.
    Note that candidates sentences which are uknown (in term of embeddings) will be removed from the candidates.
    :param embedding_distrib: embedding distributor see @EmbeddingDistributor
    :param inp_rpr: input text representation see @InputTextObj
    :return: A tuple of two element containing 1) the list of candidate sentences
    2) a numpy array of shape (number of candidate sentences, dimension of embeddings :
    each row is the embedding of one candidate sentence
    """
    candidates = np.array(extract_sent_candidates(inp_rpr))
    embeddings = np.array(embedding_distrib.get_tokenized_sents_embeddings(candidates))

    valid_candidates_mask = ~np.all(embeddings == 0, axis=1)
    return candidates[valid_candidates_mask], embeddings[valid_candidates_mask, :]
