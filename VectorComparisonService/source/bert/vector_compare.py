#!/usr/bin/python3
# Copyright (c) 2019-present, InoviaGroup, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# --------------------------------------------------------
#
# tools for vector comparison with BERT

#import logging
import numpy as np


def ComputeScoreBERT(doc_vecs, query_vec, topk=3): #, questions
    # compute normalized dot product as score
    score_1 = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score_1)[::-1][:topk]
    #create an array and return it
    #topk_scoreIDs = []
    #topk_QuestIDs = []
    #for idx in topk_idx:
        #topk_scoreIDs.append(score_1[idx])
        #topk_QuestIDs.append(questions[idx])
        #print('> %s\t%s' % (score_1[idx], questions[idx]))
    
    return topk_idx.tolist()  # topk_scoreIDs, topk_QuestIDs