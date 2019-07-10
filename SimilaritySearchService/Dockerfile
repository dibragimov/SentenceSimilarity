FROM python:3.7-stretch

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN pip install flask
RUN pip install requests
RUN pip install numpy
#### Make port 7007 available to the world outside this container
EXPOSE 7007
#### create a folder where language training files will be stored
RUN mkdir /opt/LASER
RUN mkdir /opt/LASER/GlueService
#### copying files
COPY . /opt/LASER/GlueService
#### run file
WORKDIR /opt/LASER
CMD ["python", "/opt/LASER/GlueService/flaskGlue.py", "--files-dir", "/opt/LASER/GlueService/data", "--lang", "en sv no", "--embed-service", "http://embedding:7005", "--compare-service", "http://comparison:7006", "--use-bert", "--contain-classes"]
#### "--lang-agnostic"
