from spacy import displacy as dsp
import en_core_web_lg
nlp = en_core_web_lg.load()

nytimes = nlp("He weighs a ")
entities=[(i, i.label_, i.label) for i in nytimes.ents]
print(entities)


dsp.render(nytimes, style ='ent', jupyter = True)
