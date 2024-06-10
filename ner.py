import spacy
import json

# Load SpaCy model
nlp = spacy.load("en_core_web_trf")

# Function to extract named entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    return entities

# Read text from file
with open('data/text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Extract entities
entities = extract_entities(text)

# Save entities to JSON file
with open('data/entities.json', 'w', encoding='utf-8') as json_file:
    json.dump(entities, json_file, ensure_ascii=False, indent=4)

print("Entities extracted and saved to entities.json")
