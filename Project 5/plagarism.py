import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(doc1, doc2):
    documents = [doc1, doc2]

    # Create the Document Term Matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Calculate the cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Return the similarity between the first and second document
    return similarity_matrix[0, 1]

doc1 = "This is a sample document to check for plagiarism."
doc2 = "This document is a sample used to check for plagiarism."

similarity = calculate_similarity(doc1, doc2)
print(f"Similarity: {similarity:.2f}")

if similarity > 0.8:
    print("Plagiarism detected!")
else:
    print("Documents are sufficiently different.")
