#!/usr/bin/python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Embed sentences

import os
import sys
#import argparse
#import pdb
#import faiss
#import numpy as np

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/tools')
from embed import SentenceEncoder
from text_processing import TokenLine, BPEfastApplyLine, BPEfastLoad


##########################################################################




def embedLine(line, encoder, loaded_bpe, lang='en'):
    #print("Tokenizing")
    tokenized = TokenLine(line, lang=lang, lower_case=True, romanize=False)
    #print("Finished tokenizing")
    bpeApplied = BPEfastApplyLine(tokenized, loaded_bpe)
    #print("Embedding stage following!!")
    embedded = encoder.encode_sentences([bpeApplied])
    #print(embedded)

    return embedded[0]



##########################################################################

'''
# options for encoder
parser.add_argument('--encoder', type=str, required=True,
    help='encoder to be used')
parser.add_argument(
    '--lang', '-L', nargs='+', default=None,
    help="List of languages to test on")
parser.add_argument('--buffer-size', type=int, default=10000,
    help='Buffer size (sentences)')
parser.add_argument('--max-tokens', type=int, default=12000,
    help='Maximum number of tokens to process in a batch')
parser.add_argument('--max-sentences', type=int, default=None,
    help='Maximum number of sentences to process in a batch')
parser.add_argument('--cpu', action='store_true',
    help='Use CPU instead of GPU')

args = parser.parse_args()

print('LASER: embedding single sentence')
'''

if __name__ == '__main__':
    model_dir = LASER + "/models"
    encoder = model_dir + "/bilstm.93langs.2018-12-26.pt"
    bpe_codes = model_dir + "/93langs.fcodes"
    enc = SentenceEncoder(encoder, cpu=True)
    loaded_bpe=BPEfastLoad('',bpe_codes)

    line = 'Testing to encode line'
    print("Embedding line", line)
    
    embedded = embedLine(line,enc,loaded_bpe)
    print("Finished Embedding")
    print("Embedded line len=", len(embedded))
