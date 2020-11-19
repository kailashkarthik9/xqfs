import argparse
import os
import re
import warnings
from abc import ABC, abstractmethod

import spacy

from query_generation.keyphrase_extraction.fileIO import read_file, write_string
from query_generation.keyphrase_extraction.stop_words import STOP_WORDS


class PosTagging(ABC):
    @abstractmethod
    def pos_tag_raw_text(self, text, as_tuple_list=True):
        """
        Tokenize and POS tag a string
        Sentence level is kept in the result :
        Either we have a list of list (for each sentence a list of tuple (word,tag))
        Or a separator [ENDSENT] if we are requesting a string by putting as_tuple_list = False
        Example :
        >>from sentkp.preprocessing import postagger as pt
        >>pt = postagger.PosTagger()
        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.')
        [
            [('Write', 'VB'), ('your', 'PRP$'), ('python', 'NN'),
            ('code', 'NN'), ('in', 'IN'), ('a', 'DT'), ('.', '.'), ('py', 'NN'), ('file', 'NN'), ('.', '.')
            ],
            [('Thank', 'VB'), ('you', 'PRP'), ('.', '.')]
        ]
        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)
        'Write/VB your/PRP$ python/NN code/NN in/IN a/DT ./.[ENDSENT]py/NN file/NN ./.[ENDSENT]Thank/VB you/PRP ./.'
        >>pt = postagger.PosTagger(separator='_')
        >>pt.pos_tag_raw_text('Write your python code in a .py file. Thank you.', as_tuple_list=False)
        Write_VB your_PRP$ python_NN code_NN in_IN a_DT ._. py_NN file_NN ._.
        Thank_VB you_PRP ._.
        :param as_tuple_list: Return result as list of list (word,Pos_tag)
        :param text:  String to POS tag
        :return: POS Tagged string or Tuple list
        """

        pass

    def pos_tag_file(self, input_path, output_path=None):

        """
        POS Tag a file.
        Either we have a list of list (for each sentence a list of tuple (word,tag))
        Or a file with the POS tagged text
        Note : The jumpline is only for readibility purpose , when reading a tagged file we'll use again
        sent_tokenize to find the sentences boundaries.
        :param input_path: path of the source file
        :param output_path: If set write POS tagged text with separator (self.pos_tag_raw_text with as_tuple_list False)
                            If not set, return list of list of tuple (self.post_tag_raw_text with as_tuple_list = True)
        :return: resulting POS tagged text as a list of list of tuple or nothing if output path is set.
        """

        original_text = read_file(input_path)

        if output_path is not None:
            tagged_text = self.pos_tag_raw_text(original_text, as_tuple_list=False)
            # Write to the output the POS-Tagged text.
            write_string(tagged_text, output_path)
        else:
            return self.pos_tag_raw_text(original_text, as_tuple_list=True)

    def pos_tag_and_write_corpora(self, list_of_path, suffix):
        """
        POS tag a list of files
        It writes the resulting file in the same directory with the same name + suffix
        e.g
        pos_tag_and_write_corpora(['/Users/user1/text1', '/Users/user1/direct/text2'] , suffix = _POS)
        will create
        /Users/user1/text1_POS
        /Users/user1/direct/text2_POS
        :param list_of_path: list containing the path (as string) of each file to POS Tag
        :param suffix: suffix to append at the end of the original filename for the resulting pos_tagged file.
        """
        for path in list_of_path:
            output_file_path = path + suffix
            if os.path.isfile(path):
                self.pos_tag_file(path, output_file_path)
            else:
                warnings.warn('file ' + output_file_path + 'does not exists')


class PosTaggingSpacy(PosTagging):
    """
        Concrete class of PosTagging using StanfordPOSTokenizer and StanfordPOSTagger
    """

    def __init__(self, nlp=None, separator='|', lang='en_core_web_lg'):
        if not nlp:
            print('Loading Spacy model')
            self.nlp = spacy.load(lang, entity=False)
            print('Spacy model loaded ' + lang)
        else:
            self.nlp = nlp
        self.separator = separator
        self.stop_words = STOP_WORDS

    def pos_tag_raw_text(self, text, as_tuple_list=True):
        """
            Implementation of abstract method from PosTagging
            @see PosTagging
        """

        # This step is not necessary int the stanford tokenizer.
        # This is used to avoid such tags :  ('      ', 'SP')
        text = re.sub('[ ]+', ' ', text).strip()  # Convert multiple whitespaces into one

        doc = self.nlp(text)
        if as_tuple_list:
            return [[(token.text, token.tag_) for token in sent] for sent in doc.sents]
        return '[ENDSENT]'.join(
            ' '.join(self.separator.join([token.text, token.tag_]) for token in sent) for sent in doc.sents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write POS tagged files, the resulting file will be written'
                                                 ' at the same location with _POS append at the end of the filename')

    parser.add_argument('tagger', help='which pos tagger to use [stanford, spacy, corenlp]')
    parser.add_argument('listing_file_path', help='path to a text file '
                                                  'containing in each row a path to a file to POS tag')
    args = parser.parse_args()

    pt = PosTaggingSpacy()
    suffix = 'SPACY'
    list_of_path = read_file(args.listing_file_path).splitlines()
    print('POS Tagging and writing ', len(list_of_path), 'files')
    pt.pos_tag_and_write_corpora(list_of_path, suffix)
