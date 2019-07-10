from flask import Flask
from flask import jsonify
from flask import request
import traceback
import logging
from source.laser.vector_compare import SearchIndex, BuildIndex
from source.bert.vector_compare import ComputeScoreBERT
import numpy as np


#### maybe use Redis for future load balancing?????????
#### a dictionary storing lang and all related embeddings for FAISS
base_embeddings = {} 
current_Is = {}
#### a dictionary storing lang and all related embeddings in BERT
bert_embeddings = {} 

app = Flask(__name__)


@app.route('/baseupload/<string:lang>', methods=['POST'])
def uploadSentences(lang):
    """Updates the array of embeddings with new embeddings
    Creates an endpoint that takes in a numpy array of floats in json format 
    and appends these embeddings to the list it keeps.
    The input JSON format should be:
    {
        "embedding":"<embedding>"
        "isbert": True/False ---> optional parameter. Default is False.
    }
    """
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response
    data = request.get_json()
    embeddings = data['embedding']
    isBERT = False
    if 'isbert' in data:
        isBERT = data['isbert'] == True
    #logging.info("Passed vector: {}".format(embeddings))
    try:
        vec_size = -1
        if isBERT: #### BERT comparison
            global bert_embeddings
            if bert_embeddings.get(lang) is None:
                bert_embeddings[lang] = np.array(embeddings)
            else:
                bert_embeddings[lang] = np.concatenate(
                        (bert_embeddings[lang], np.array(embeddings))
                        )
            vec_size = bert_embeddings[lang].shape[0]
            #logging.warning("The shape of the array is {}".format(
            #        bert_embeddings[lang].shape))
        else: #### LASER/FAISS comparison
            global base_embeddings
            global current_Is
            if base_embeddings.get(lang) is None: #### questions by this lang ID do not exist
                 base_embeddings[lang] = embeddings
            else:  #### merge with existing questions
                base_embeddings[lang] = base_embeddings[lang] + embeddings
            #### Build the index after the addition
            vec_size, current_Is[lang] = BuildIndex(base_embeddings[lang])
            logging.warning("FAISS index size {}".format(vec_size))
        
        #logging.info("Finished uploading")
        response = jsonify({"vectorsize": vec_size})
        response.status_code = 200
    except Exception as e:
        #print(e)
        response = jsonify(
            {"message": "An error occured while embedding the sentence.",
            "exception":str(e)})
        logging.error("exception {}, stacktrace {}".format(str(e), 
                      str(traceback.format_exc())))
        response.status_code = 500

    return response


@app.route('/compare/<string:lang>', methods=['POST'])
def compareVector(lang):
    """compares a vector with existing embeddings to find similar vectors
    Creates an endpoint that takes in a numpy array of floats in json format 
    and compares this vector to previously uploaded vectors.
    The input JSON format should be:
    {
        "embedding":"<embedding>"
        "topk":"<topk>" ---> optional parameter. Default is 3
        "isbert": True/False ---> optional parameter. Default is False.
    }
    """
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response
    data = request.get_json()
    embedding = data['embedding']
    #### check if is BERT
    isBERT = False
    if 'isbert' in data:
        isBERT = data['isbert'] == True
    #### check how many answers to return
    topk = 3
    if 'topk' in data:
        topk = data['topk'] 
    try:
        results = []
        if isBERT:
            global bert_embeddings
            results = ComputeScoreBERT(bert_embeddings[lang], 
                                       np.array(embedding), topk)
        else:
            global current_Is
            #### do the comparison and return index
            results = SearchIndex(embedding, current_Is[lang], topk)
        #logging.info('Result {}'.format(results))
        response = jsonify({"results": results})
        response.status_code = 200
    except Exception as e:
        #print(e)
        response = jsonify(
            {"message": "An error occured while searching for results.",
            "exception ": str(e)})
        logging.error("exception {}, stacktrace {}".format(str(e), 
                      str(traceback.format_exc())))
        response.status_code = 500

    return response

@app.route('/clearbase/<string:lang>', methods=['POST'])
def clearSentences(lang):
    """Sends a command to clear the array of embeddings for specific language
    Creates an endpoint that takes one argument in json format 
    and clears existing embeddings.
    The input JSON format should be:
    {
        "isbert": True/False ---> optional parameter. Default is False.
    }
    """
    logging.warning(request)
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response
    data = request.get_json()
    isBERT = False
    if 'isbert' in data:
        isBERT = data['isbert'] == True
    #logging.info("Passed vector: {}".format(embeddings))
    try:
        vec_size = -1
        if isBERT : #### BERT comparison
            global bert_embeddings
            if not (bert_embeddings.get(lang) is None):
                bert_embeddings[lang] = None
            #logging.warning("The shape of the array is {}".format(
            #        bert_embeddings[lang].shape))
        else: #### LASER/FAISS comparison
            global base_embeddings
            global current_Is
            #if not (base_embeddings.get(lang) is None): #### questions by this lang ID do not exist
            #     base_embeddings[lang] = None
            #### Build the index after the addition
            base_embeddings[lang] = None
            current_Is[lang] = None
            
        if not isBERT and base_embeddings.get(lang) is None and \
        current_Is.get(lang) is None:
            vec_size = 0
        elif isBERT and bert_embeddings.get(lang) is None:
            vec_size = 0
        
        #logging.info("Finished uploading")
        response = jsonify({"vectorsize": vec_size})
        response.status_code = 200
    except Exception as e:
        #print(e)
        response = jsonify(
            {"message": "An error occured while embedding the sentence.",
            "exception":str(e)})
        logging.error("exception {}, stacktrace {}".format(str(e), 
                      str(traceback.format_exc())))
        response.status_code = 500

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', use_reloader=True, debug=True, port=7006)
