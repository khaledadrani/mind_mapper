import json

import stanza

import stanza


# Download the English language model (you only need to run this once)
stanza.download('en')


nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
doc = nlp("Chris Manning teaches at Stanford University. He lives in the Bay Area.")
print(*[f'entity: {ent.text}\ttype: {ent.type}' for sent in doc.sentences for ent in sent.ents], sep='\n')

# Sample text
text = "Barack Obama was born in Hawaii."

with open('data/text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Process the text
doc = nlp(text)

# Initialize list to store entities
entities_list = []

# Extract named entities
for sentence in doc.sentences:
    for entity in sentence.ents:
        entities_list.append({"entity": entity.text, "type": entity.type})

# Save entities to JSON file
with open("data/stanza_named_entities.json", "w") as json_file:
    json.dump(entities_list, json_file, indent=4)
