import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import words
import random

nltk.download('words')
vocabulary = [word.lower() for word in words.words() if len(word) == 5]
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 1))
word_vectors = vectorizer.fit_transform(vocabulary).toarray()
word_map = {word: vec for word, vec in zip(vocabulary, word_vectors)}

def compute_similarity(word, candidates):
    if word not in word_map:
        raise ValueError(f"Word '{word}' is not in the vocabulary.")
    word_vec = word_map[word].reshape(1, -1)
    candidate_vecs = np.array([word_map[c] for c in candidates])
    if candidate_vecs.size == 0:
        raise ValueError("No valid candidates to compare.")
    similarities = cosine_similarity(word_vec, candidate_vecs).flatten()
    return dict(zip(candidates, similarities))

def eliminate_candidates(candidates, guess, feedback):
    updated_candidates = []
    for word in candidates:
        valid = True
        for i, char in enumerate(guess):
            if feedback[i] == "G" and word[i] != char:
                valid = False
            elif feedback[i] == "Y" and (char not in word or word[i] == char):
                valid = False
            elif feedback[i] == "X" and char in word:
                valid = False
        if valid:
            updated_candidates.append(word)
    return updated_candidates

if __name__ == "__main__":
    candidates = vocabulary
    current_guess = random.choice(candidates)
    print(f"First Guess: {current_guess}")
    while True:
        feedback = input("Enter feedback (X for gray, Y for yellow, G for green): ").upper()
        if len(feedback) != 5 or any(c not in "XYG" for c in feedback):
            print("Invalid feedback. Please enter a string of 5 characters using X, Y, G.")
            continue
        if feedback == "GGGGG":
            print(f"The program successfully guessed the word: {current_guess}")
            break
        candidates = eliminate_candidates(candidates, current_guess, feedback)
        print(f"Remaining Candidates: {len(candidates)}")
        if candidates:
            similarities = compute_similarity(current_guess, candidates)
            print("Top 5 Similarities for Remaining Candidates:")
            top_5 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
            for word, similarity in top_5:
                print(f"{word}, cosine similarity: {similarity:.4f}")
            current_guess = max(top_5, key=lambda x: x[1])[0]
            print("Next Guess:", current_guess)
        else:
            print("No remaining candidates. Something went wrong.")
            break
