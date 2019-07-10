from flask import Flask
from flask import jsonify
from flask import request
from source.helper import loadQuestionsFromFile, retrieveEmbeddings
from source.helper import uploadQuestionsToService, loadTextAndClassesFromFiles
import argparse
import requests
import traceback
import logging
import os
import time

app = Flask(__name__)
embed_service = ''#'http://127.0.0.1:7005'
compare_service = ''#'http://127.0.0.1:7006'
is_bert_supported = False
class_support = False
lang_agnostic = False
lang_questions = {}
all_questions = []
lang_classes = {}
all_classes = []

@app.route('/compare/<string:lang>', methods=['POST'])
def compareSentence(lang):
    """compares a sentence with existing sentences to find similar sentences
    Creates an endpoint that takes in a sentence in json format
    and compares this sentence to previously uploaded sentences (from file).
    The input JSON format should be:
    {
        "sentence":"<sentence>"
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
    sentence = data['sentence']
    topk = 3
    if 'topk' in data:
        topk = int(data['topk'])
    isbert = False
    if 'isbert' in data:
        isbert = data['isbert'] == True
    try:
        #### do the comparison and return index
        embedding = retrieveEmbeddings(embed_service, sentence, lang, isbert)
        #url string
        urlstr = compare_service.strip('/') + '/compare'
        urlstr = urlstr + '/' + lang
        input_text = {'embedding': embedding, 'topk': topk}
        if isbert:
            input_text.update(isbert=True )
        result_response = requests.post(urlstr, json=input_text).json()
        #logging.info('Result {}'.format(result_response))
        #### now retrieve texts from list
        results = []
        ids = result_response['results']
        global lang_questions
        for i in ids:
            sntns=lang_questions.get(lang)[i]
            results.append(sntns)
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

@app.route('/reload/<string:lang>', methods=['POST'])
def reloadBase(lang):
    """accepts a list of sentences to be a new base for comparison
    Creates an endpoint that takes in a list of sentences in json format
    and reinitializes a vector base. BERT support is enabled by CLI argument
    The input JSON format should be:
    {
        "sentences":"[sentences]"
    }
    """
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response
    data = request.get_json()
    sentences = data['sentences']
    global is_bert_supported
    try:
        #### do the comparison and return index
        laser_embeddings = retrieveEmbeddings(embed_service, sentences, lang)
        #url string to clear current base questions
        urlstr = compare_service.strip('/') + '/clearbase'
        urlstr = urlstr + '/' + lang
        input_text = {'isbert': False}
        clear_resp = requests.post(urlstr, json=input_text).json()
        v_size = clear_resp['vectorsize']
        if int(v_size) != 0:
            logging.warning("Vectorsize is {}, System is LASER".format(v_size))
        
        if is_bert_supported:
            bert_embeddings = retrieveEmbeddings(embed_service, sentences, lang, is_bert_supported)
            input_text = {'isbert': is_bert_supported}
            clear_resp = requests.post(urlstr, json=input_text).json()
            v_size = clear_resp['vectorsize']
            if int(v_size) != 0:
                logging.warning("Vectorsize is {}, System is BERT".format(v_size))
        
        #### now upload texts to VectorComparison
        global lang_questions
        lang_questions[lang] = sentences
        logging.warning("number of questions for {} is {}".format(lang, len(lang_questions[lang])))
        #tot_size = uploadQuestionsToService(compare_service,
        #                                    embedding, lang)
        tot_size = uploadQuestionsToService(compare_service,
                                            laser_embeddings, lang)
        if is_bert_supported:
            tot_size = uploadQuestionsToService(compare_service,
                                            bert_embeddings, lang, 
                                            is_bert_supported)
            
        logging.warning(
                    "{:d} questions for {} language were loaded to the system".format(
                    tot_size, lang))
        response = jsonify({"vectorsize": tot_size})
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

@app.route('/compareclass/<string:lang>', methods=['POST'])
def compareClassSentence(lang):
    """compares a sentence with existing sentences to find similar sentences
    Creates an endpoint that takes in a sentence in json format
    and compares this sentence to previously uploaded sentences (from file).
    Also returns the classes the answers belong to
    The input JSON format should be:
    {
        "sentence":"<sentence>"
        "topk":"<topk>" ---> optional parameter. Default is 3
        "isbert": True/False ---> optional parameter. Default is False.
    }
    the output JSON format is:
    {
        "results":"<sentences>"
        "classes":"<classes for the sentences>" ---> number depends on topk
    }
    """
    if not request.is_json:
        response = jsonify(
            {"message": "ERROR. The mime type needs to be application/json"})
        response.status_code = 415
        return response
    data = request.get_json()
    sentence = data['sentence']
    topk = 3
    if 'topk' in data:
        topk = int(data['topk'])
    isbert = False
    if 'isbert' in data:
        isbert = data['isbert'] == True
    ids = []
    try:
        #### do the comparison and return index
        embedding = retrieveEmbeddings(embed_service, sentence, lang, isbert)
        #url string
        urlstr = compare_service.strip('/') + '/compare'
        urlstr = urlstr + '/' + lang
        input_text = {'embedding': embedding, 'topk': topk}
        if isbert:
            input_text.update(isbert=True )
        result_response = requests.post(urlstr, json=input_text).json()
        #logging.info('Result {}'.format(result_response))
        #### now retrieve texts from list
        results = []
        classes = []
        ids = result_response['results']
        global lang_questions
        global lang_classes
        global all_questions
        global all_classes
        if not lang_agnostic: 
            for i in ids:
                sntns = lang_questions.get(lang)[i]
                clss = lang_classes.get(lang)[i]
                results.append(sntns)
                classes.append(clss)
        else:
            for i in ids:
                sntns = all_questions[i]
                clss = all_classes[i]
                results.append(sntns)
                classes.append(clss)
        response = jsonify({"results": results, "classes": classes})
        response.status_code = 200
    except Exception as e:
        #print(e)
        response = jsonify(
            {"message": "An error occured while searching for results.",
            "exception ": str(e)})
        logging.error("exception {}, stacktrace {}, IDs {}".format(str(e),
                      str(traceback.format_exc()), ids))
        response.status_code = 500

    return response

#### initializes the service by loading files from the directory 
#### and using them as a base for further comparison
def init(files_dir, langs):
    time.sleep(30) ##### sleep 30 seconds to allow BERT to be up and running
    global lang_questions
    global all_questions
    if not class_support:
        lang_questions, all_questions = loadQuestionsFromFile(files_dir, langs)
    else:
        global lang_classes
        lang_questions, lang_classes = loadTextAndClassesFromFiles(files_dir, 
                                                                   langs)
    for lang in langs:
        if lang_questions.get(lang) is not None:
            embeddings = retrieveEmbeddings(embed_service, lang_questions.get(lang), lang)
            tot_size = uploadQuestionsToService(compare_service, embeddings, lang)
            logging.warning(
                    "{:d} questions for {} language were loaded to the system for LASER".format(
                    tot_size, lang))
            if is_bert_supported:
                embeddings_bert = retrieveEmbeddings(embed_service,
                                            lang_questions.get(lang), 
                                            lang, is_bert_supported)
                tot_size = uploadQuestionsToService(compare_service,
                                                embeddings_bert, lang, 
                                                is_bert_supported)
                logging.warning(
                    "{:d} questions for {} language were loaded to the system for BERT".format(
                    tot_size, lang))

#### initializes the service by loading files from the directory 
#### and using them as a base for further comparison
#### Language Agnostic - all sentences treated as the same language - English ('en')
def langAgnosticInit(files_dir, langs):
    time.sleep(30) ##### sleep 30 seconds to allow BERT to be up and running
    global lang_questions
    global all_questions
    if not class_support:
        lang_questions, all_questions = loadQuestionsFromFile(files_dir, langs)
    else:
        global lang_classes
        lang_questions, lang_classes = loadTextAndClassesFromFiles(files_dir, 
                                                                   langs)
    global all_classes
    for lang in langs:
        if lang_questions.get(lang) is not None:
            all_questions = all_questions + list(lang_questions.get(lang))
            all_classes = all_classes + list(lang_classes.get(lang))
            
            
    embeddings = retrieveEmbeddings(embed_service, all_questions, 'en')
    tot_size = uploadQuestionsToService(compare_service, embeddings, 'en')
    logging.warning("{:d} questions for {} language were loaded to the system for LASER".format(
                    tot_size, 'general'))
    if is_bert_supported:
        embeddings_bert = retrieveEmbeddings(embed_service,
                                            all_questions, 
                                            'en', is_bert_supported)
        tot_size = uploadQuestionsToService(compare_service,
                                            embeddings_bert, 'en', 
                                            is_bert_supported)
        logging.warning(
                    "{:d} questions for {} language were loaded to the system for BERT".format(
                    tot_size, 'general'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Similarity: similarity service')
    parser.add_argument('--embed-service', type=str, required=True,
                        help='URL of the embedding service')
    parser.add_argument('--compare-service', type=str, required=True,
                        help='URL of the vector comparison service')
    parser.add_argument('--files-dir', type=str, required=True,
                        help='Directory where the original text files with questions are located')
    parser.add_argument('--lang', type=str, required=True,
                        help='List of language files to load. Example "en ch de" ')
    parser.add_argument('--use-bert', required=False, action='store_true',
                        help='Flag to include BERT support')
    parser.add_argument('--contain-classes', required=False, action='store_true',
                        help='Flag to specify whether files contain classes')
    parser.add_argument('--lang-agnostic', required=False, action='store_true',
                        help='Flag to specify whether all files should be treated as containing general (English) language for classification')
    
    args, unknown = parser.parse_known_args()
    files_dir = args.files_dir ####'/home/dilshod/Documents/Scripts/...'
    embed_service = args.embed_service #'http://127.0.0.1:7005'
    compare_service = args.compare_service #'http://127.0.0.1:7006/'
    langs = args.lang.split(' ')
    if args.use_bert:
        is_bert_supported = True
        logging.warning("BERT support enabled")
    if args.contain_classes:
        class_support = True
        logging.warning("System is set to support classes")
    if args.lang_agnostic:
        lang_agnostic = True
        logging.warning("System is set to be language agnostic (Use 'en' for all languages)")
    if not (str(os.environ.get("WERKZEUG_RUN_MAIN")).strip() != 'true'):
        logging.info('Running initialization')
        if not lang_agnostic:
            init(files_dir, langs)
        else:
            langAgnosticInit(files_dir, langs)
        logging.info('Finished initialization')
    
    app.run(host='0.0.0.0', use_reloader=True, debug=True, port=7007)
    
    
