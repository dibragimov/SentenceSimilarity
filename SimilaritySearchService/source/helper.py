# -*- coding: utf-8 -*-

import requests
import os
import logging

def loadQuestionsFromFile(file_dir, langs):
    onlyfiles = [f for f in os.listdir(file_dir) if os.path.isfile(
            os.path.join(file_dir, f)) and os.path.splitext(f)[1][1:] in langs]
    logging.warning("# of files loaded {}".format(len(onlyfiles)))
    #### A dictionary containing langID and questions in that lang
    lang_questions = {} 
    all_questions = [] #### a list with all questions independent of language
    for file in onlyfiles:
        logging.warning("File to process {}".format(file))
        currentLang = os.path.splitext(os.path.join(file_dir, file))[1][1:]
        with open(os.path.join(file_dir, file), encoding='utf-8', 
                  errors='surrogateescape') as f:
            texts = f.read().splitlines()#f.readlines()
            logging.info(' -   {:s}: {:d} lines'.format(file, len(texts)))
            all_questions = all_questions + [
                    (lambda x: x.strip())(x) for x in texts
                    ] #texts #### add all the questions to the list
            if lang_questions.get(currentLang) is None: 
                #### questions by this lang ID do not exist
                lang_questions[currentLang] = texts
            else:
                #### merge with existing questions
                lang_questions[currentLang] = lang_questions[currentLang] + texts 
            f.close()
        
    return lang_questions, all_questions

def retrieveEmbeddings(embed_service, sentences, lang, isbert=False):
    urlstr = embed_service.strip('/') + '/embed'
    urlstr = urlstr + '/' + lang
    input_text  = {'sentences':sentences}
    if isbert:
        input_text.update(isbert = True)
    response = requests.post(urlstr, json=input_text).json()
    base_embs = response['embedding']
    return base_embs

def uploadQuestionsToService(compare_service, embeddings, lang, isbert=False):
    urlstr = compare_service.strip('/') + '/baseupload'
    urlstr = urlstr + '/' + lang
    input_text  = {'embedding':embeddings}
    if isbert:
        input_text.update(isbert = True)
    response = requests.post(urlstr, json=input_text).json()
    total_size = response['vectorsize']
    #logging.info("totalsize {}".format(total_size))
    return total_size

#### pass the list with lines containing class and question
#### returns 2 lists - one list with all classes and one list with all questions
def getTextAndClassesFromList(q_list):
    current_quests = [ (lambda x: x.strip().split(';', 1))(x) for x in q_list ]
    classes = [quest[0] for quest in current_quests ]
    questions = [quest[1] for quest in current_quests ]
    return classes, questions 

#### load classes of answers and the answers themselves
#### the line has structure "classID;sentence"
def loadTextAndClassesFromFiles(file_dir, langs):
    #### files are like ecommerce_en.csv
    #### os.path.splitext(f)[0][-2:] - gets the lang part
    #### os.path.splitext(f)[1][1:] == 'csv' - check if csv
    onlyfiles = [f for f in os.listdir(file_dir) if os.path.isfile(
            os.path.join(file_dir, f)) and os.path.splitext(f)[0][-2:] in langs
                and os.path.splitext(f)[1][1:] == 'csv']
    logging.warning("# of files loaded {}".format(len(onlyfiles)))
    #### A dictionary containing langID and questions in that lang
    lang_questions = {}
    #### A dictionary containing langID and classes in that lang
    lang_classes = {}
    for file in onlyfiles:
        logging.warning("File to process {}".format(file))
        currentLang = os.path.splitext(os.path.join(file_dir, file))[0][-2:]
        with open(os.path.join(file_dir, file), encoding='utf-8', 
                  errors='surrogateescape') as f:
            texts = f.read().splitlines()
            logging.warning(' -   {:s}: {:d} lines'.format(file, len(texts)))
            ####
            curr_classes, curr_questions = getTextAndClassesFromList(texts)
            #
            if lang_questions.get(currentLang) is None: 
                #### questions by this lang ID do not exist
                lang_questions[currentLang] = curr_questions
            else:
                #### merge with existing questions
                lang_questions[currentLang] = lang_questions[currentLang] + curr_questions 
            if lang_classes.get(currentLang) is None: 
                #### classes by this lang ID do not exist
                lang_classes[currentLang] = curr_classes
            else:
                #### merge with existing classes
                lang_classes[currentLang] = lang_classes[currentLang] + curr_classes 
            f.close()
    return lang_questions, lang_classes