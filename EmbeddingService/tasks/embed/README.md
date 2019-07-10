# LASER: calculation of sentence embeddings

This codes shows how to calculate sentence embeddings for
an arbitrary text file:
```
bash ./embed.sh INPUT-FILE LANGUAGE OUTPUT-FILE
```
The input will be tokenized, using the mode of the specified language, BPE will be applied
and the sentence embeddings will be calculated.

## Output format

The embeddings are stored in float32 matrices in raw binary format.
They can be read in Python by:
```
import numpy as np
dim = 1024
X = np.fromfile("my_embeddings.raw", dtype=np.float32, count=-1)                                                                          
X.resize(X.shape[0] // dim, dim)                                                                                                 
```
X is a N x 1024 matrix where N is the number of lines in the text file.
        
## Example
```
./embed.sh ${LASER}/data/tatoeba/v1/tatoeba.fra-eng.fra fr my_embeddings.raw
```

## Added code for Flask service

The code in `embedSentence.py` has been added to easily embed a singe line. This one is used by the `flaskEmbed.py` code and it would be suboptimal to use this script or the Flask-endpoint to embed a large set of sentences, for example a training dataset. 

To embed sentences from a training/test/dev-dataset it is better to build and run the docker image on some server and use the bash-script described above to calculate the embeddings. After this is done, you can use your embeddings to train and evaluate a model. 