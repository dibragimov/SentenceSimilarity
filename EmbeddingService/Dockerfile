FROM python:3.6-stretch

# install dependencies
# RUN apt-get update --fix-missing 
RUN apt-get install -y wget bzip2 ca-certificates curl unzip 

#### install some prerequisites
RUN pip install --upgrade pip
RUN pip install numpy

########RUN git clone https://github.com/facebookresearch/LASER.git

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

# ========= install pytorch =========

RUN pip install --upgrade pip
RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip install torch torchvision

# ========= END ===================

# ========= LASER ============
WORKDIR /opt

COPY /source /opt/LASER/source
COPY /tasks /opt/LASER/tasks
COPY install_external_tools.sh /opt/LASER
COPY install_models.sh /opt/LASER
COPY flaskEmbed.py /opt/LASER

# ======= Install FAISS =========

RUN conda install faiss-cpu -c pytorch


WORKDIR /opt/LASER

ENV LASER /opt/LASER
RUN bash ./install_models.sh
RUN bash ./install_external_tools.sh

# ========= END ==============

# ======== Install PIP dependencied =========
COPY requirements.txt /opt/LASER
RUN pip install -r /opt/LASER/requirements.txt
RUN pip install bert-serving-client
# ========= END ==============

#### Make port 7005 available to the world outside this container
EXPOSE 7005

WORKDIR /opt/LASER
#### run file
CMD ["python", "/opt/LASER/flaskEmbed.py", "--bert-service", "bertembedding"]
