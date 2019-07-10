FROM python:3.7-stretch

# install dependencies
# RUN apt-get update --fix-missing 
RUN apt-get install -y wget bzip2 ca-certificates curl unzip 

#### ========= install some prerequisites =========
RUN pip install numpy
RUN pip install tensorflow
RUN pip install bert-serving-server  # server
## ========= END ===================

#### Make port 7001,7002 available to the world outside this container
EXPOSE 7001 7002

#### =========  create a folder where language training files will be downloaded. get it and extract =========
RUN mkdir /opt/BERT
RUN mkdir /opt/BERT/L_768_A_12
WORKDIR /opt/BERT
RUN wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

# ========= TEMPORARY ============ 
#COPY uncased_L-12_H-768_A-12.zip /opt/BERT
# ========= END ==============

RUN unzip -q -o uncased_L-12_H-768_A-12.zip -d /opt/BERT/L_768_A_12 
RUN mv L_768_A_12/uncased_L-12_H-768_A-12/* -t L_768_A_12/
RUN rm /opt/BERT/uncased_L-12_H-768_A-12.zip
RUN rmdir L_768_A_12/uncased_L-12_H-768_A-12
## ========= END ===================

#### run file
CMD ["bert-serving-start", "-model_dir", "/opt/BERT/L_768_A_12", "-max_seq_len=40", "-max_batch_size=16", "-port=7001", "-port_out=7002"]
