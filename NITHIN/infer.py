from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
nlp2 = spacy.load("/home/nithing/model_dir_P_D(1)")
doc2 = nlp2("director,nithin g,producer ,hashim andulla,jismon lukose,charles david")
for ent in doc2.ents:
    print(ent.label_, ent.text)
