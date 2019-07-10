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

import faiss
import logging
import numpy as np


###############################################################################
# create an FAISS index on the given vectors/embeddings (embedding_arr)

def IndexCreate(embedding_arr, idx_type,
                verbose=False, normalize=True, save_index=False, dim=1024):

    assert idx_type == 'FlatL2', 'only FlatL2 index is currently supported'
    x = np.array(embedding_arr, dtype="float32")
    nbex = x.shape[0] // dim
    logging.info(' - embedding: size {:d} {:d} examples of dim {:d}'
          .format(len(embedding_arr), nbex, dim))
    x.resize(nbex, dim)
    logging.info(' - creating FAISS index')
    logging.info(x)
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
    idx.add(x)
    if save_index:
        iname = 'TODO'
        logging.info(' - saving index into ' + iname)
        faiss.write_index(idx, iname)
    return x, idx


###############################################################################
# search and return indexes of closest vectors for passed vector

def SearchIndex(emb_vector, idx, topk=3):
    embedding = np.array(emb_vector, dtype='float32')
    dim = 1024
    nbex = embedding.shape[0] // dim    #### number of questions
    #### to reshape to correct size     #embedding = embedding.reshape(nbex,dim) 
    embedding.resize(nbex,dim)
    results = []
    D, I = idx.search(embedding, topk)
    logging.info('Index {}'.format(I))
    for j in range(topk): #### convert it to int - later json can't 
        #### convert numpy datatypes
        results.append(int(I[0][j])) #### 0 because we have only one question
        #### - if we have list of questions - we can return list of list with
        #### loop for I[i in range(nbex)][j]
    return results


###############################################################################
# build FAISS index and return its length

def BuildIndex(embeddings):
    D, I = IndexCreate(embeddings, 'FlatL2')
    return int(D.shape[0]), I


