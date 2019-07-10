FROM python:3.7-stretch

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8


# ========= install dependencies ========= 
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    build-essential \
    curl \
    git-core screen \
    htop \
    pkg-config \
    unzip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh


WORKDIR /opt

ENV PATH /opt/conda/bin:$PATH

# ========= install pytorch =========

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip install torch torchvision

# ======= Install FAISS =========

RUN conda install faiss-cpu -c pytorch


# =========  Make port 7006 available to the world outside this container  ========= 
EXPOSE 7006

# ========= create a folder and copy files ==============
WORKDIR /opt/LASER
RUN mkdir /opt/LASER/VectorCompareService
COPY . /opt/LASER/VectorCompareService
#### run file
CMD ["python", "/opt/LASER/VectorCompareService/flaskCompare.py"]
