# this is a quick script to review built-in spacy document similarity

# imports
import spacy
import numpy as np
from spacy import cli
from scipy.spatial.distance import pdist, squareform

# load up the spacy model
model = "en_core_web_md"
cli.download(model)
nlp = spacy.load(model) 

# create a few documents
doc1 = nlp("Brock likes to play golf")
doc2 = nlp("Tiger woods is an excellent golfer")
doc3 = nlp("Python is an amazing programming language for analytics and data science.")

# get the vectors and make numpy array
dvs = [doc1.vector, doc2.vector, doc3.vector]
dvs = np.array(dvs)

# what do we have
dvs.shape

# lets calcualte the cosine distance between each of the 3 documents 
dist = squareform(pdist(dvs, metric="cosine"))
dist

# spacy actually tries to make this easier for us
# docs have a similarity method
doc1.similarity(doc2)

# NOTE:  spacy is similarity, what we calculated is distance
# lets just invert it
1- dist