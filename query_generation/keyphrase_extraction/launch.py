from query_generation.keyphrase_extraction.emb_distrib_local import EmbeddingDistributorLocal
from query_generation.keyphrase_extraction.input_representation import InputTextObj
from query_generation.keyphrase_extraction.method import MMRPhrase
from query_generation.keyphrase_extraction.postagging import PosTaggingSpacy
import pickle


def extract_keyphrases(embedding_distrib, ptagger, idf_model, raw_text, N, lang, beta=0.55, alias_threshold=0.7):
    """
    Method that extract a set of keyphrases
    :param embedding_distrib: An Embedding Distributor object see @EmbeddingDistributor
    :param ptagger: A Pos Tagger object see @PosTagger
    :param idf_model: A TFIDFVectorizer object see @PosTagger
    :param raw_text: A string containing the raw text to extract
    :param N: The number of keyphrases to extract
    :param lang: The language
    :param beta: beta factor for MMR (tradeoff informativness/diversity)
    :param alias_threshold: threshold to group candidates as aliases
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """
    tagged = ptagger.pos_tag_raw_text(raw_text)
    text_obj = InputTextObj(tagged, lang)
    return MMRPhrase(embedding_distrib, text_obj, raw_text, ptagger, idf_model, N=N, beta=beta, alias_threshold=alias_threshold)


def load_local_embedding_distributor(sent2vec_model_path):
    # config_parser = ConfigParser()
    # config_parser.read('config.ini')
    # sent2vec_model_path = config_parser.get('SENT2VEC', 'model_path')
    return EmbeddingDistributorLocal(sent2vec_model_path)


if __name__ == '__main__':
    raw_text = "This is Lionel Messi's house"
    n = 5
    embedding_distributor = load_local_embedding_distributor('../../models/sent2vec/wiki_bigrams.bin')
    idf_model = pickle.load(open('../../models/idf/idf_cnndm.pkl', 'rb'))
    pos_tagger = PosTaggingSpacy()
    print(extract_keyphrases(embedding_distributor, pos_tagger, idf_model, raw_text, n, 'en'))
