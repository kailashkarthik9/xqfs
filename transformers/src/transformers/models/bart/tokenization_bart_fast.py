# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
from transformers import add_start_docstrings

from .tokenization_bart import BartTokenizer
from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
from ...tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING, BatchEncoding
from ...utils import logging

logger = logging.get_logger(__name__)

# vocab and merges same as roberta
vocab_url = "https://huggingface.co/roberta-large/resolve/main/vocab.json"
merges_url = "https://huggingface.co/roberta-large/resolve/main/merges.txt"
tokenizer_url = "https://huggingface.co/roberta-large/resolve/main/tokenizer.json"
_all_bart_models = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "yjernite/bart_eli5",
    # This is not exhaustive: see https://huggingface.co/models?filter=bart
]

import spacy


class BartTokenizerFast(RobertaTokenizerFast):
    # merges and vocab same as Roberta
    max_model_input_sizes = {m: 1024 for m in _all_bart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_bart_models},
        "merges_file": {m: merges_url for m in _all_bart_models},
        "tokenizer_file": {m: tokenizer_url for m in _all_bart_models},
    }
    slow_tokenizer_class = BartTokenizer

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
            self,
            src_texts: List[str],
            tgt_texts: Optional[List[str]] = None,
            max_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            padding: str = "longest",
            return_tensors: Optional[str] = None,
            truncation=True,
            **kwargs,
    ) -> BatchEncoding:
        if max_length is None:
            max_length = self.model_max_length
        model_inputs: BatchEncoding = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs


class QfsBartTokenizerFast(RobertaTokenizerFast):
    # merges and vocab same as Roberta
    max_model_input_sizes = {m: 1024 for m in _all_bart_models}
    pretrained_vocab_files_map = {
        "vocab_file": {m: vocab_url for m in _all_bart_models},
        "merges_file": {m: merges_url for m in _all_bart_models},
        "tokenizer_file": {m: tokenizer_url for m in _all_bart_models},
    }
    slow_tokenizer_class = BartTokenizer

    def __init__(self, vocab_file, merges_file, **kwargs):
        super().__init__(vocab_file, merges_file, **kwargs)
        self.nlp = spacy.load('en_core_web_sm')

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
            self,
            src_texts: List[str],
            tgt_texts: Optional[List[str]] = None,
            max_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
            padding: str = "longest",
            return_tensors: Optional[str] = None,
            truncation=True,
            **kwargs,
    ) -> BatchEncoding:
        if max_length is None:
            max_length = self.model_max_length
        model_inputs: BatchEncoding = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            return_offsets_mapping=True,
            truncation=truncation,
            **kwargs,
        )
        model_inputs['query_relevance_ids'] = self.add_query_relevance_ids(src_texts, model_inputs['offset_mapping'])
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        labels = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    def add_query_relevance_ids(self, src_texts, offsets_mapping):
        batch_query_relevance_ids = list()
        for text, offsets in zip(src_texts, offsets_mapping):
            text = text.lower()
            query = text.split('[q]')[0].strip()
            query_lemmas = {token.lemma_ for token in self.nlp(query)}
            query_relevance_spans = self.get_query_relevance_spans(text, query_lemmas)
            query_relevance_ids = self.get_query_relevance_ids(offsets, query_relevance_spans)
            batch_query_relevance_ids.append(query_relevance_ids)
        return torch.LongTensor(batch_query_relevance_ids)

    def get_query_relevance_spans(self, text, query_lemmas):
        doc = self.nlp(text)
        query_tokens = list()
        for token in doc:
            if token.lemma_ in query_lemmas:
                query_tokens.append({
                    'token': token.text,
                    'start': token.idx,
                    'end': token.idx + len(token.text),
                })
        query_relevance_spans = list()
        for token in query_tokens:
            query_relevance_spans.extend(range(token['start'], token['end']))
        return set(query_relevance_spans)

    @staticmethod
    def get_query_relevance_ids(offsets, query_relevance_spans):
        query_relevance_ids = list()
        for offset in offsets:
            if len(query_relevance_spans.intersection(range(offset[0].item(), offset[1].item()))) > 0:
                query_relevance_ids.append(1)
            else:
                query_relevance_ids.append(0)
        return query_relevance_ids
