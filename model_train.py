# import spacy

# # Load the medium or large model for better accuracy
# nlp = spacy.load("en_core_web_md")

# # Example: Extract named entities
# doc = nlp("John has 5 years of experience in Python and Data Science.")
# entities = [(ent.text, ent.label_) for ent in doc.ents]
# print(entities)



import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json


nlp = spacy.load("en_core_web_lg") # load a new spacy model
db = DocBin() # create a DocBin object
    
f = open('resumeparse.json',encoding='utf-8')
TRAIN_DATA = json.load(f)
    
    
for text, annot in tqdm(TRAIN_DATA['annotations']):
    print(text,annot)
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./training_data.spacy")



# put base config file and run this code{ python -m spacy init fill-config base_config.cfg config.cfg}



# import spacy
# import PyPDF2 



# def pdf_to_text(pdf_file_path):

#     with open(pdf_file_path, 'rb') as pdf_file:

#         pdf_reader = PyPDF2.PdfReader(pdf_file)

#         text = ""

#         for page_num in range(len(pdf_reader.pages)):

#             page = pdf_reader.pages[page_num]

#             text += page.extract_text()

#     return text



# # Example usage:

# pdf_path = "resume.pdf" 

# text_data = pdf_to_text(pdf_path)


# doc=nlp(text)
# for ent in doc.ents:
#     print(ent.text,ent.label_)













# print(text_data) 


# # Load the medium model for better accuracy
# nlp = spacy.load("en_core_web_md")

# # # Provide the path to your text file
# # file_path = "2.txt"

# # # Read the content of the text file
# # with open(file_path, 'r', encoding='utf-8') as file:
# #     text = file.read()

# # Process the text using SpaCy
# doc = nlp(text_data)

# # Extract named entities
# entities = [ ent.label_ for ent in doc.ents]
# print("Named Entities:", entities)
