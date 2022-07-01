from bs4 import BeautifulSoup
import requests
import pandas as pd
URL = 'https://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html'

page = requests.get(URL, verify=False)
soup = BeautifulSoup(page.content, 'lxml')

diseases = []
occurences = []
symptoms = []
prev = ""
prev_occ = 0
idx = 0


def clean(value):
    simple = []
    last = value.split('_')
    here = last[-1]
    simple.append(here)
    for i in range(0, len(simple)):
        here = simple[i]
        disease = here.replace('\n', "")
        disease = disease.replace("  ", " ")
        disease.strip()
        return disease


for row in soup.table.find_all('tr'):
    col = row.find_all('td')
    all_disease = row.td.get_text()
    if(len(all_disease) > 3):
        disease = clean(all_disease)
        occurence = (col[1].get_text())
        symptom = (col[2].get_text())
        symptom = clean(symptom)
        prev = disease
        prev_occ = occurence
    else:
        disease = prev
        occurence = prev_occ
        symptom = (col[2].get_text())
        symptom = clean(symptom)

    if(idx != 0):
        diseases.append(disease)
        occurences.append(occurence)
        symptoms.append(symptom)

    idx += 1

print(occurences[0])
dictionary = {'disease': diseases,
              'symptom': symptoms, 'occurence': occurences}
df = pd.DataFrame(dictionary)
to_csv = df.to_csv('C:/Users/DELL/OneDrive/Desktop/New folder/disease.csv')
