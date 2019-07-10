from flask import Flask
from flask import jsonify
from flask import request
from source.embed import SentenceEncoder
from tasks.embed.embedSentence import embedLine
from source.lib.text_processing import BPEfastLoad
from source.embed_bert import encodeSentencesBERT
import os
import argparse
import traceback
import logging
import numpy as np

# get environment
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

bert_service = None
port=7001
port_out=7002

# Initiates the sentence encoder
# encoder
model_dir = LASER + "/models"
encoder = model_dir + "/bilstm.93langs.2018-12-26.pt"
bpe_codes = model_dir + "/93langs.fcodes"

enc = SentenceEncoder(encoder, cpu=True)
loaded_bpe=BPEfastLoad('',bpe_codes)

app = Flask(__name__)


@app.route('/embed/<string:lang>', methods=['POST'])
def embedSentence(lang):
    """Returns embedded sentences
    Creates an endpoint that takes in a lis of sentences in json format and 
    returns embeddings done by using LASER/BERT.
    The input JSON format should be:
    {
        "sentences":"<sentences>"
        "isbert": True/False ---> optional parameter. Default is False
    }
    """
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response
    data = request.get_json()
    sentences = data['sentences']
    isBERT = False
    if 'isbert' in data:
        isBERT = data['isbert'] == True
    try:
        embeddings = None
        #print("Sentences:",sentences)
        if isBERT == True: #### Use BERT embedding
            ip = 'localhost'
            if not(bert_service is None):
                ip = bert_service
            """logging.info(
                    "BERT encoding. port: {}, port_out {}, ipaddr {}".format(
                            port, port_out, ip)
                    )"""
            embeddings = encodeSentencesBERT(sentences, ipaddr=ip, port=port, 
                                         port_out=port_out)
        else: #### Use LASER embedding
            for sentence in sentences:
                embedding = embedLine(sentence, encoder=enc, 
                                      loaded_bpe=loaded_bpe, lang=lang)
                #logging.info('semntence: {}, embedding: {}'.format(sentence, embedding))
                if embeddings is None: #### if first sentence - assign to returned array
                    embeddings = embedding
                else: #### concatenate two numpy arrays returning all embeddings as one array
                    embeddings = np.concatenate((embeddings, embedding), 
                                                axis=None)
        #logging.info("Finished embeddings")
        response = jsonify({"embedding": embeddings.tolist()})
        response.status_code = 200
    except Exception as e:
        #print(e)
        response = jsonify(
            {"message": "An error occured while embedding the sentence.",
            "exception": str(e)})
        logging.warning("exception {}, stacktrace {}".format(str(e), 
                        str(traceback.format_exc())))
        response.status_code = 500

    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Similarity: Embedding service')
    parser.add_argument('--bert-service', type=str, required=False,
                        help='Server name of the BERT-AS-A-SERVICE')
    parser.add_argument('--bert-port', type=int, required=False,
                        help='Port number for BERT-AS-A-SERVICE')
    parser.add_argument('--bert-port-out', type=int, required=False,
                        help='Output port number for BERT-AS-A-SERVICE')
    
    args, unknown = parser.parse_known_args()
    if args.bert_service:
        bert_service = args.bert_service #'127.0.0.1'
    if args.bert_port:
        port = args.bert_port
    if args.bert_port_out:
        port_out = args.bert_port_out
    logging.info("BERT details - Host: {} Port:{} Port_out:{}".format(bert_service, port, port_out))
    app.run(host='0.0.0.0', use_reloader=True, debug=True, port=7005)
