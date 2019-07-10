
import argparse
import datetime
import logging
import numpy as np
import os
import random
import requests
from sklearn.model_selection import train_test_split

dflt_base_dir = '/home/dilshod/Documents/Scripts/'
service_url = ''

#### load QUORA questions from the file
def loadQuoraQuestions(file_dir, start_pos, number_of_questions):
    k = 0 #### line number in a file
    if number_of_questions == 0:
        number_of_questions = 405000 ### max 404000 lines in a file
    #### A dictionary containing questionID and questionText
    loaded_questions = {} 
    #### a list with IDs containing a tuple with 2 values - similar questions
    loaded_similar_q = [] 
    with open(os.path.join(file_dir, 'quora_duplicate_questions.tsv'),
                  encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            try:
                if k == 0:
                    k = k+1
                    continue #### skip the first line
                k = k+1
                if start_pos > k: #### skip the line to start from start_pos
                    continue
                values = line.split('\t')
                #### fill in the dictionary
                if not(values[1] in loaded_questions): 
                    loaded_questions[int(values[1])] = values[3]
                if not(values[2] in loaded_questions):
                    loaded_questions[int(values[2])] = values[4]
                #### fill in similar question list
                if int(values[5]) == 1: 
                    loaded_similar_q.append([int(values[1]), int(values[2])]) 
                 #### loaded needed lines - stop reading
                if k > (number_of_questions + start_pos):
                    break
            except: 
                #### for test skipping a couple of lines does not matter
                print("An exception occurred in line "+str(k)+": "+line)
        f.close()
    return loaded_questions, loaded_similar_q


#### get all the IDs of questions similar to the current question
def getSimilarQuestions(question_id, similar_q_list):
    similars = []
    for i in range(len(similar_q_list)):
        if similar_q_list[i][0] == question_id: #### check if the ID is in first or second column
            similars.append(similar_q_list[i][1])
        elif similar_q_list[i][1] == question_id:
            similars.append(similar_q_list[i][0])
        
    return similars


#### get all questions from the Quora dataset for specified IDs
#### questions - dictionary with key (id) and value (the text of the question)
def getQuestionsFromIDList(id_list, questions): 
    q_list = []
    for i in id_list:
        q_list.append(str(questions.get(i)).strip())
    return q_list

#### based on the text - return the ID of the question from the dictionary
#### questions - dictionary with key (id) and value (the text of the question)
def getIDFromQuestionText(q_text, questions):  #### questions - entries of a dictionary
    for key, value in questions.items():
        if value == q_text:
            return key
        

##### saving questions in the list to the file 
def saveTextToFile(questions, filename):
    with open(filename, 'w') as f:
        for item in questions:
            f.write("%s\n" % item)
        f.close()
        
#### saving experiment results to a file 
def saveResultToFile(file_dir, num_total_questions, size_id, 
                     num_train_questions, num_test_questions, k, isbert,
                     result): 
    ##### write to file
    now = datetime.datetime.now()

    filename = 'results_{}.txt'.format(now.strftime("%Y-%m-%d"))

    if os.path.exists(os.path.join(file_dir, filename)):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not


    results_file = open(os.path.join(file_dir, filename), append_write)
    if append_write == 'w' : #### column names
        results_file.write("total_q,size_id,train_q,test_q,topK,isbert,correct%\n")
    # write results
    results_file.write(
        "{},{},{},{},{},{},{}\n".format(num_total_questions, size_id,
         num_train_questions, num_test_questions, k, isbert, result) 
    )
    results_file.close()
    
#### do one round of experiments - save data to a file
def runExperiment(quora_file_dir, k, num_questions, start_pos):
    questions = {} #### A dictionary containing questionID and questionText
    similar_q = [] #### a list with IDs containing a tuple with 2 values - similar questions
    ################### load random questions from QUORA dataset
    questions, similar_q = loadQuoraQuestions(quora_file_dir, 
                                              start_pos, num_questions)

    num_total_questions = len(questions)
    num_paraph_questions = len(set(np.array(similar_q).flatten()))
    logging.warning('Number of questions loaded: {}, number of questions with paraphrases {}'
          .format(str(num_total_questions), num_paraph_questions))
    
    #### need to split only questions that have paraphrases
    #### otherwise testing part will not contain values that have paraphrased counterparts
    train_question_ids, test_question_ids = train_test_split(
            list(set(np.array(similar_q).flatten())), test_size=0.3)
    ##### include in training all other questions except those from testset
    questions_test = getQuestionsFromIDList(test_question_ids, questions)
    questions_train = getQuestionsFromIDList(
            (set(questions.keys()-set(test_question_ids))), questions)
    
    num_train_questions = len(questions_train)
    num_test_questions = len(questions_test)
    logging.warning(
            "Number of questions in TestSet: {}, in TrainSet: {}".format(
                    num_test_questions, num_train_questions))
    #### load train questions to the service
    global service_url
    urlstr = service_url.strip('/') + '/reload'
    urlstr = urlstr + '/' + 'en'
    logging.warning(urlstr)
    input_text = {'sentences': questions_train}
    input_text.update(isbert=True)
#    logging.warning(input_text)
    clear_resp = requests.post(urlstr, json=input_text).json()
    if int(clear_resp['vectorsize']) != 0:
        logging.warning("loaded questions number: {}".format(int(clear_resp['vectorsize'])))
    #### run experiment
    found_correctly_laser = []
    found_correctly_bert = []
    for q in questions_test:
        #### get the answer from the service
        laser_answer = askServer(service_url, q, topk=k, isbert=False)
        bert_answer = askServer(service_url, q, topk=k, isbert=True)
        real_similar_questions= getQuestionsFromIDList(
                getSimilarQuestions(
                        getIDFromQuestionText(q, questions), similar_q), 
                        questions)
        logging.warning(
                "Real question: {} \nLASER answers {} \nBERT answers {} \nReal answers {}".format(
                        q, laser_answer, bert_answer, real_similar_questions))
        for i in range(k):
            curr_laser_answers = laser_answer[0:i+1]
            curr_bert_answers = bert_answer[0:i+1]
            #### initializa
            if len(found_correctly_laser) < i+1:
                found_correctly_laser.insert(i, [])
                logging.warning('initialization of found_correctly_laser[{}]'.format(i))
            if len(found_correctly_bert) < i+1:
                found_correctly_bert.insert(i, [])
                logging.warning('initialization of found_correctly_bert[{}]'.format(i))
            ####
            if set(curr_laser_answers) & set(real_similar_questions): #quora_q_id_test in quora_q_ids_training - set intersection:
                found_correctly_laser[i].append(1) #### correct answer
            else:
                found_correctly_laser[i].append(0) #### correct answer is not insed the returned answers
                
            if set(curr_bert_answers) & set(real_similar_questions): #quora_q_id_test in quora_q_ids_training - set intersection:
                found_correctly_bert[i].append(1) #### correct answer
            else:
                found_correctly_bert[i].append(0) #### correct answer is not insed the returned answers
#    logging.warning('found_correctly_laser {}'.format(found_correctly_laser))
#    logging.warning('found_correctly_bert {}'.format(found_correctly_bert))
    for i in range(k):
        result_laser = round(float(sum(found_correctly_laser[i]))/len(found_correctly_laser[i]), 5) #### round it to 5 numbers after the comma
        saveResultToFile(quora_file_dir, num_total_questions, num_questions, 
                         num_train_questions, num_test_questions, i+1, False, result_laser)
        result_bert = round(float(sum(found_correctly_bert[i]))/len(found_correctly_bert[i]), 5) #### round it to 5 numbers after the comma
        saveResultToFile(quora_file_dir, num_total_questions, num_questions, 
                         num_train_questions, num_test_questions, i+1, True, result_bert)

def askServer(ulraddr, sentence, topk, isbert):
    input_text  = {'sentence':[sentence], 'topk': topk}
    input_text.update(isbert=isbert)
    urlstr = service_url.strip('/') + '/compare'
    urlstr = urlstr + '/' + 'en'
    question_response = requests.post(urlstr, json=input_text).json()
    logging.warning("askServer answer {}".format(question_response))
    return question_response['results'] if 'results' in question_response else None

#### entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Similarity TEST')
    parser.add_argument('--quora-file-dir', type=str, required=True,
                        help='Direcory where the original quora file with questions is downloaded')
    parser.add_argument('--service-url', type=str, required=True,
                        help='Direcory where the original quora file with questions is downloaded')
    #### get values from command line interface
    args, unknown = parser.parse_known_args()
    quora_file_dir = args.quora_file_dir ####r'/home/dilshod/Downloads'
    service_url = args.service_url
    #### do it for top 1, 2, 3
    topK = 3 
    #### experiment with these numbers 5 times
    limits = [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000] 
    #### run it 5 times with different set of questions
    for i in range(len(limits)):
        for x in range(5):
            startpos = random.randrange(404000 - limits[i])
            #print('starting from : '+str(startpos))
            runExperiment(quora_file_dir, topK, limits[i], startpos)