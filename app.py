import os
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_md')


student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_files.remove("requirements.txt")
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]


def preprocess(text):
    set(spacy.lang.en.stop_words.STOP_WORDS)
    return ' '.join([token.text.lower() for token in nlp(text) if not token.is_stop])


def vectorize(Text):
    vectors = [nlp(preprocess(text)).vector for text in Text]
    return np.array(vectors)


def similarity(doc1, doc2):
    return cosine_similarity([doc1], [doc2])[0][0]


vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()


def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

plagiarism_results = check_plagiarism()

for data in plagiarism_results:
    if data[2] > 0.8:
        with open(data[0], 'r', encoding='utf-8') as file_a:
            text_a = file_a.read()
        with open(data[1], 'r', encoding='utf-8') as file_b:
            text_b = file_b.read()
        print(f"Similarity score between {data[0]} and {data[1]}: {data[2]:.2f}")
        print("Text similarity:")
        print("="*50)
        print(text_a)
        print("="*50)
        print(text_b)
        print("="*50)
        print("\n\n")


print("======== Documents Similarity ========")
for data in plagiarism_results:
    print(data)
