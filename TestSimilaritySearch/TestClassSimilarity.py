import argparse
import datetime
import logging
import numpy as np
import os
import requests

#### pass the list with lines containing class and question
#### returns 2 lists - one list with all classes and one list with all questions
def getTextAndClassesFromList(q_list):
    current_quests = [ (lambda x: x.strip().split(';', 1))(x) for x in q_list ]
    classes = [quest[0] for quest in current_quests ]
    questions = [quest[1] for quest in current_quests ]
    return classes, questions 

##### load classes of answers and the answers themselves
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

#### saving experiment results to a file 
def saveResultToFile(file_dir, lang, k, isbert, result): 
    ##### write to file
    now = datetime.datetime.now()

    filename = 'results_class_{}.txt'.format(now.strftime("%Y-%m-%d"))

    if os.path.exists(os.path.join(file_dir, filename)):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not


    results_file = open(os.path.join(file_dir, filename), append_write)
    if append_write == 'w' : #### column names
        results_file.write("lang,topK,isbert,correct%\n")
    # write results
    results_file.write(
        "{},{},{},{}\n".format(lang, k, isbert, result) 
    )
    results_file.close()

#### method to run compare with class information and to report back the results
def askClassServer(ulraddr, sentence, topk, isbert, lang='en'):
    input_text  = {'sentence':[sentence], 'topk': topk}
    input_text.update(isbert=isbert)
    urlstr = service_url.strip('/') + '/compareclass'
    urlstr = urlstr + '/' + lang #### for now mostly English
    question_response = requests.post(urlstr, json=input_text).json()
    logging.warning("askClassServer answer {}".format(question_response))
    if 'results' in question_response:
        return question_response['results'], question_response['classes']
    else:
        return None, None


#### do one round of experiments - save data to a file
def runExperiment(file_dir, k, langs):
    questions = {} #### A dictionary containing questionText for each lang
    classes = {} #### a dict with class IDs for specific questions
    ################### load random questions from QUORA dataset
    questions, classes = loadTextAndClassesFromFiles(file_dir, langs)

    num_total_questions = len(list(questions.values()))
    logging.warning('Number of questions loaded for testing: {}'
          .format(str(num_total_questions)))
    
    #### run experiment
    found_correctly_laser = []
    found_correctly_bert = []
    for l in langs:
        lang_questions = questions.get(l)
        lang_classes = classes.get(l)
        ind = 0
        for q in lang_questions:
            #### get the answer from the service
            laser_answers, laser_classes = askClassServer(service_url, q, topk=k, isbert=False, lang=l)
            bert_answers, bert_classes = askClassServer(service_url, q, topk=k, isbert=True, lang=l)
            real_question_class = lang_classes[ind]
            logging.warning(
                    "Real question: {} \nLASER answers {} \nBERT answers {} \nReal answer {}".format(
                            q, laser_classes, bert_classes, real_question_class))
            for i in range(k):
                curr_laser_answers = laser_classes[0:i+1]
                curr_bert_answers = bert_classes[0:i+1]
                #### initializa
                if len(found_correctly_laser) < i+1:
                    found_correctly_laser.insert(i, [])
                    logging.warning('initialization of found_correctly_laser[{}]'.format(i))
                if len(found_correctly_bert) < i+1:
                    found_correctly_bert.insert(i, [])
                    logging.warning('initialization of found_correctly_bert[{}]'.format(i))
                ####
                if real_question_class in curr_laser_answers: #quora_q_id_test in quora_q_ids_training - set intersection:
                    found_correctly_laser[i].append(1) #### correct answer
                else:
                    found_correctly_laser[i].append(0) #### correct answer is not insed the returned answers
                    
                if real_question_class in curr_bert_answers: #quora_q_id_test in quora_q_ids_training - set intersection:
                    found_correctly_bert[i].append(1) #### correct answer
                else:
                    found_correctly_bert[i].append(0) #### correct answer is not insed the returned answers
            
            ind = ind + 1

        for i in range(k):
            result_laser = round(float(sum(found_correctly_laser[i]))/len(found_correctly_laser[i]), 5) #### round it to 5 numbers after the comma
            saveResultToFile(file_dir, l, i+1, False, result_laser)
            result_bert = round(float(sum(found_correctly_bert[i]))/len(found_correctly_bert[i]), 5) #### round it to 5 numbers after the comma
            saveResultToFile(file_dir, l, i+1, True, result_bert)


#### do one round of experiments for lang agnostic test - save data to a file
def runLangAgnosticExperiment(file_dir, k, langs):
    questions = {} #### A dictionary containing questionText for each lang
    classes = {} #### a dict with class IDs for specific questions
    ################### load random questions from QUORA dataset
    questions, classes = loadTextAndClassesFromFiles(file_dir, langs)

    num_total_questions = len(questions)
    logging.warning('Number of questions loaded for testing: {}'
          .format(str(num_total_questions)))
    
    #### run experiment
    found_correctly_laser = []
    found_correctly_bert = []
    for l in langs:
        lang_questions = questions.get(l)
        lang_classes = classes.get(l)
        ind = 0
        for q in lang_questions:
            #### get the answer from the service
            laser_answers, laser_classes = askClassServer(service_url, q, topk=k, isbert=False)
            bert_answers, bert_classes = askClassServer(service_url, q, topk=k, isbert=True)
            real_question_class = lang_classes[ind]
            logging.warning(
                    "Real question: {} \nLASER answers {} \nBERT answers {} \nReal answer {}".format(
                            q, laser_classes, bert_classes, real_question_class))
            for i in range(k):
                curr_laser_answers = laser_classes[0:i+1]
                curr_bert_answers = bert_classes[0:i+1]
                #### initializa
                if len(found_correctly_laser) < i+1:
                    found_correctly_laser.insert(i, [])
                    logging.warning('initialization of found_correctly_laser[{}]'.format(i))
                if len(found_correctly_bert) < i+1:
                    found_correctly_bert.insert(i, [])
                    logging.warning('initialization of found_correctly_bert[{}]'.format(i))
                ####
                if real_question_class in curr_laser_answers: #quora_q_id_test in quora_q_ids_training - set intersection:
                    found_correctly_laser[i].append(1) #### correct answer
                else:
                    found_correctly_laser[i].append(0) #### correct answer is not insed the returned answers
                    
                if real_question_class in curr_bert_answers: #quora_q_id_test in quora_q_ids_training - set intersection:
                    found_correctly_bert[i].append(1) #### correct answer
                else:
                    found_correctly_bert[i].append(0) #### correct answer is not insed the returned answers
            
            ind = ind + 1

        for i in range(k):
            result_laser = round(float(sum(found_correctly_laser[i]))/len(found_correctly_laser[i]), 5) #### round it to 5 numbers after the comma
            saveResultToFile(file_dir, l, i+1, False, result_laser)
            result_bert = round(float(sum(found_correctly_bert[i]))/len(found_correctly_bert[i]), 5) #### round it to 5 numbers after the comma
            saveResultToFile(file_dir, l, i+1, True, result_bert)


#### entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Similarity TEST')
    parser.add_argument('--test-file-dir', type=str, required=True,
                        help='Direcory where the original testing file with questions is downloaded')
    parser.add_argument('--service-url', type=str, required=True,
                        help='URL of the running Glue service')
    parser.add_argument('--lang', type=str, required=True,
                        help='List of language files to load. Example "en ch de" ')
    #### get values from command line interface
    args, unknown = parser.parse_known_args()
    file_dir = args.test_file_dir ####r'/home/dilshod/Downloads'
    service_url = args.service_url
    langs = args.lang.split(' ')
    #### do it for top 1, 2, 3
    topK = 3 
    #### experiment with these files
    runExperiment(file_dir, topK, langs)
    #runLangAgnosticExperiment(file_dir, topK, langs)