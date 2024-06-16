import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from scipy.stats import mode


def normalize(x_i):
    """
    Normalize the vector
    """
    min_i = tf.math.reduce_min(x_i)
    max_i = tf.math.reduce_max(x_i)
    x_i_normalized = (x_i - min_i) / (max_i - min_i)
    return x_i_normalized

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if (norm_vec1 * norm_vec2) != 0 else 0

def rank_by_similarity(query_vector, database_vectors):
    """
    Rank the database vectors by their cosine similarity to the query vector
    and return the ranking indices along with the cosine similarity scores.
    """
    similarities = [cosine_similarity(query_vector, db_vec) for db_vec in database_vectors]
    ranked_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    sorted_similarities = np.array(similarities)[ranked_indices]
    return ranked_indices, sorted_similarities

def evaluate_test_vectors(test_vectors, train_vectors, y_test, y_train):
    """
    Ranks each test vector against all train vectors by similarity.

    :param test_vectors: np.array, The test feature vectors.
    :param train_vectors: np.array, The train feature vectors.
    :param y_test: np.array, The labels for the test vectors.
    :param y_train: np.array, The labels for the train vectors.
    """
    # Number of test vectors
    num_test_vectors = test_vectors.shape[0]

    # Iterate over each test vector and rank it against the train vectors.
    for i in range(num_test_vectors):
        query_vector = test_vectors[i]
        ranked_indices, cosine_scores = rank_by_similarity(query_vector, train_vectors)

        print(f"Query Vector {i+1}:")
        print("Ranked order of database indices:", ranked_indices)
        print("Ranked train vector labels:", [y_train[ind] for ind in ranked_indices])
        print("Ranked order of database indices cosine similarity:", cosine_scores)
        print("\n" + "-"*50 + "\n")

def evaluate_test_vectors_to_csv(test_vectors, train_vectors, y_test, y_train, csv_filename):
    output_data = []
    num_test_vectors = test_vectors.shape[0]

    for i in range(num_test_vectors):
        query_vector = test_vectors[i]
        ranked_indices, cosine_scores = rank_by_similarity(query_vector, train_vectors)

        # Update output_data with the current test vector's results
        for rank, index in enumerate(ranked_indices):
            output_data.append({
                "Test Vector Index": i + 1,
                "Train Vector Rank": rank + 1,
                "Train Vector Index": index,
                "Train Vector Label": y_train[index],
                "Cosine Similarity": cosine_scores[rank]
            })

        print(f"Query Vector {i+1}:")
        print("Ranked order of database indices:", ranked_indices)
        print("Ranked train vector labels:", [y_train[ind] for ind in ranked_indices])
        print("Ranked order of database indices cosine similarity:", cosine_scores)
        print("\n" + "-"*50 + "\n")

    # Create DataFrame and Save to CSV
    df = pd.DataFrame(output_data)
    df.to_csv(csv_filename, index=False)
    print(f"Output saved to {csv_filename}")

def evaluate_test_vectors_with_counterfactuals(test_vectors, train_vectors, y_test, y_train, KT_LOSO_Full_Train):
    """
    Evaluates test vectors against training vectors using 1-NN and K-NN approaches and finds counterfactuals.

    :param test_vectors: np.array, The test feature vectors.
    :param train_vectors: np.array, The train feature vectors.
    :param y_test: np.array, The labels for the test vectors.
    :param y_train: np.array, The labels for the train vectors.
    :param k_values: list, Values of K for K-NN evaluation.
    """
    num_test_vectors = test_vectors.shape[0]
    predictions_1nn = []
    predictions_1nn_trial_names = []
    #predictions_knn = {k: [] for k in k_values}
    predictions_knn = []
    counterfactuals_1nn = []
    counterfactual_1nn_trial_names = []

    # Evaluate each test vector
    for i in range(num_test_vectors):
        query_vector = test_vectors[i]
        ranked_indices, _ = rank_by_similarity(query_vector, train_vectors)

        # 1-NN Prediction and Counterfactual
        prediction_1nn = y_train[ranked_indices[0]]
        predictions_1nn.append(prediction_1nn)

        prediction_1nn_trial_name = KT_LOSO_Full_Train[ranked_indices[0]] # factual
        predictions_1nn_trial_names.append(prediction_1nn_trial_name)

        counterfactual_1nn = next((idx for idx in ranked_indices if y_train[idx] != prediction_1nn), None)
        counterfactuals_1nn.append(counterfactual_1nn)

        counterfactual_1nn_trial_name = KT_LOSO_Full_Train[counterfactual_1nn] # counterfactual
        counterfactual_1nn_trial_names.append(counterfactual_1nn_trial_name)

        # 3-NN Predictions 
        top_3_labels = [y_train[ind] for ind in ranked_indices[:3]]
        mode_result = mode(top_3_labels)
        predicted_label = mode_result.mode[0]
        predictions_knn.append(predicted_label)

    print("Predictions 1NN: ", predictions_1nn)
    print("Prediction/Factual Trial Names: ", predictions_1nn_trial_names)
    print("Predictions 3-NN: ", predictions_knn)
    print("counterfactuals 1NN: ", counterfactuals_1nn)
    print("Counterfactual Trial Names: ", counterfactual_1nn_trial_names)

    return predictions_1nn, counterfactuals_1nn, predictions_knn, predictions_1nn_trial_names, counterfactual_1nn_trial_names
