from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import numpy as np
import sklearn
import nltk
from datasketch import MinHash, MinHashLSH
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = (
    r"D:\Pillais College Of Engineering\Semester VII\NPL_Project_Final\static"
)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config["UPLOAD_FOLDER"] = os.path.join(APP_ROOT, UPLOAD_FOLDER)
content = ""


def jaccard_similarity(document1, document2):
    tokens1 = nltk.word_tokenize(document1)
    tokens2 = nltk.word_tokenize(document2)

    set1 = set(tokens1)
    set2 = set(tokens2)

    jaccard_similarity = len(set1 & set2) / len(set1 | set2)
    content = "The value obtained from Jaccard similarity represents the similarity between two documents in terms of their shared elements relative to their total combined elements. It ranges from 0 (no shared elements) to 1 (identical sets). The closer the value is to 1, the more similar the documents are in terms of the elements (e.g., words or tokens) they share."
    return [jaccard_similarity, content]


def cosine_similarity_score(document1, document2):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()

    vectorizer.fit([document1, document2])

    vectors = vectorizer.transform([document1, document2]).toarray()

    cosine_similarity = np.dot(vectors[0], vectors[1]) / (
        np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
    )
    content = "The value you obtained from cosine similarity indicates the degree of similarity between the two documents. It ranges from -1 (completely dissimilar) to 1 (perfectly similar), with 0 representing no similarity. The closer the value is to 1, the more similar the documents are in terms of their content, while values closer to -1 indicate dissimilarity."

    return [cosine_similarity, content]


def hamming_distance(document1, document2):
    if len(document1) != len(document2):
        raise ValueError("documents must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(document1, document2))


def longest_common_subsequence(document1, document2):
    m, n = len(document1), len(document2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if document1[i - 1] == document2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    content = "The value obtained from Longest Common Subsequence (LCS) measures the similarity between two documents by finding the longest sequence of characters or tokens that appears in the same order within both documents. The longer the LCS value, the more similar the documents are in terms of their sequential content. A higher LCS value indicates a greater degree of similarity, while a lower value suggests dissimilarity. This metric is particularly useful for comparing sequences or textual content and can help identify common elements or patterns between documents."
    return [dp[m][n], content]


def smith_waterman(document1, document2, match=2, mismatch=-1, gap=-1):
    m, n = len(document1), len(document2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if document1[i - 1] == document2[j - 1]:
                dp[i][j] = max(
                    dp[i - 1][j - 1] + match, dp[i - 1][j] + gap, dp[i][j - 1] + gap
                )
            else:
                dp[i][j] = max(
                    dp[i - 1][j - 1] + mismatch, dp[i - 1][j] + gap, dp[i][j - 1] + gap
                )
    content = "The value obtained from the Smith-Waterman algorithm represents a similarity score that quantifies the degree of local alignment between two documents or sequences. This score indicates the strength of the local similarity between specific sections of the documents, rather than their entire content. A higher Smith-Waterman score suggests a more significant local similarity, with the specific value indicating the strength of the match."
    return [dp[m][n], content]


def needleman_wunsch(doc1, doc2, match_score=2, mismatch_score=-1, gap_penalty=-2):
    rows, cols = len(doc1) + 1, len(doc2) + 1
    score_matrix = np.zeros((rows, cols))

    for i in range(rows):
        score_matrix[i][0] = i * gap_penalty
    for j in range(cols):
        score_matrix[0][j] = j * gap_penalty

    for i in range(1, rows):
        for j in range(1, cols):
            if doc1[i - 1] == doc2[j - 1]:
                match = score_matrix[i - 1][j - 1] + match_score
            else:
                match = score_matrix[i - 1][j - 1] + mismatch_score

            delete = score_matrix[i - 1][j] + gap_penalty
            insert = score_matrix[i][j - 1] + gap_penalty

            score_matrix[i][j] = max(match, delete, insert)

    content = "The value obtained from the Needleman-Wunsch algorithm represents a similarity score that quantifies the degree of global sequence alignment between two documents or sequences. This score measures the overall similarity between the entire content of the documents. A higher Needleman-Wunsch score suggests a stronger global similarity, with the specific value indicating the strength of the match. "
    return [score_matrix[rows - 1][cols - 1], content]


def minhash_lsh_similarity(doc1, doc2, num_perm=128, threshold=0.5):
    def create_minhash(document, num_perm):
        minhash = MinHash(num_perm=num_perm)
        for word in document.split():
            minhash.update(word.encode("utf8"))
        return minhash

    minhash1 = create_minhash(doc1, num_perm)
    minhash2 = create_minhash(doc2, num_perm)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    lsh.insert("doc1", minhash1)

    result = lsh.query(minhash2)
    content = "The value obtained using MinHash and LSH represents an estimate of the Jaccard similarity between the two documents. It's a measure of how many unique elements (e.g., words or tokens) the two documents share, relative to their total unique elements. The higher the value, the more similar the documents are in terms of their content, and vice versa."

    if result:
        return [minhash1.jaccard(minhash2), content]
    else:
        return [0, content]


def bert(document1, document2):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokens1 = tokenizer(document1, return_tensors="pt", padding=True, truncation=True)
    tokens2 = tokenizer(document2, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**tokens2).last_hidden_state.mean(dim=1)
    content = "The value obtained from BERT (Bidirectional Encoder Representations from Transformers) represents a similarity score that measures the semantic similarity between the contents of two documents. BERT calculates this score by analyzing the contextual understanding of words and their relationships in the documents. Higher BERT similarity scores indicate that the documents share more semantically related content, while lower scores suggest less similarity."
    return [cosine_similarity(embeddings1, embeddings2), content]


def calculate_similarity(document1, document2, choice):
    try:
        with open(document1, "r") as file:
            file_contents1 = file.read()
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", str(e))
    try:
        with open(document2, "r") as file:
            file_contents2 = file.read()
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", str(e))
    if choice == "1":
        similarity = jaccard_similarity(file_contents1, file_contents2)
        return ["Jaccard Similarity:" + str(similarity[0]), similarity[1]]
    elif choice == "2":
        similarity = cosine_similarity_score(file_contents1, file_contents2)
        return ["Cosine Similarity:" + str(similarity[0]), similarity[1]]
    elif choice == "3":
        similarity = minhash_lsh_similarity(file_contents1, file_contents2)
        return ["Minhash LSH Similarity Score:" + str(similarity[0]), similarity[1]]
    elif choice == "4":
        similarity = longest_common_subsequence(file_contents1, file_contents2)
        return ["Longest Common Subsequence:" + str(similarity[0]), similarity[1]]
    elif choice == "5":
        similarity = smith_waterman(file_contents1, file_contents2)
        return ["Smith-Waterman Score:" + str(similarity[0]), similarity[1]]
    elif choice == "6":
        similarity = needleman_wunsch(file_contents1, file_contents2)
        return ["Needleman-Wunsch Score:" + str(similarity[0]), similarity[1]]
    elif choice == "7":
        similarity = bert(file_contents1, file_contents2)
        return ["BERT Similarity:" + str(similarity[0][0][0]), similarity[1]]


@app.route("/", methods=["POST", "GET"])
def main():
    upload = False
    if request.method == "POST":
        document1 = request.files["doc1"]
        document2 = request.files["doc2"]
        choice = request.form["similarity-measure"]

        if document1.filename == "" or document2.filename == "":
            return render_template("index.html", message=0)
        else:
            filename_1 = secure_filename(document1.filename)
            document1.save(os.path.join(app.config["UPLOAD_FOLDER"], filename_1))

            filename_2 = secure_filename(document2.filename)

            document2.save(os.path.join(app.config["UPLOAD_FOLDER"], filename_2))
            similarity_score = calculate_similarity(
                os.path.join(app.config["UPLOAD_FOLDER"], filename_1),
                os.path.join(app.config["UPLOAD_FOLDER"], filename_2),
                choice,
            )

            return render_template(
                "index.html", message=similarity_score[0], content=similarity_score[1]
            )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=8000)
