#!/usr/bin/python3
# Copyright (c) 2019-present, InoviaGroup, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# --------------------------------------------------------
#
# tools for vector comparison with FAISS

from bert_serving.client import BertClient
#import logging
#import numpy as np


###############################################################################
# create embeddings for BERT

def encodeSentencesBERT(sentences, ipaddr='127.0.0.1',
                port=5555, port_out=5556):
    
    bc = BertClient(ip=ipaddr, port=port, port_out=port_out, 
                    check_version = False)
    doc_vecs = bc.encode(sentences)
    return doc_vecs


