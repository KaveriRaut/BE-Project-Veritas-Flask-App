import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# 'Simple Ratio', 'partial Ratio', 'Token Set Ratio', 'word match percentage', 'last word match', 'character_matching_percentage', 'ngrams'
from feature_extraction import calculate_simple_ratio, calculate_partial_ratio, calculate_token_set_ratio, calculate_word_matching_percentage, last_word_match, character_matching_percentage, calculate_ngrams_similarity

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    string_features = [x for x in request.form.values()]

    # Extract features using the defined functions
    simple_ratio = calculate_simple_ratio(*string_features)
    partial_ratio = calculate_partial_ratio(*string_features)
    token_set_ratio = calculate_token_set_ratio(*string_features)
    word_match_percent = calculate_word_matching_percentage(*string_features)
    last_word_matching = last_word_match(*string_features)
    character_match_percent = character_matching_percentage(*string_features)
    ngrams = calculate_ngrams_similarity(*string_features)

    # Create a NumPy array from the extracted features
    features = np.array([[simple_ratio, partial_ratio, token_set_ratio, word_match_percent, last_word_matching, character_match_percent, ngrams]])

    # Print the received strings and features to the console
    print("Received strings:", string_features)
    print("Extracted features:", simple_ratio, partial_ratio, token_set_ratio, word_match_percent, last_word_matching, character_match_percent, ngrams)
    print("Features array:", features)

    # Make the prediction
    prediction = model.predict(features)
    
    # return render_template("index.html", features_text2=features, prediction_text="Given 2 Comapanies ["+string_features[0]+"] and ["+string_features[1]+"] are: {}".format(prediction))
    return render_template("index.html", prediction_result="Similarity Prediction: {}".format(prediction), features_text2=features, company1=string_features[0], company2=string_features[1])

if __name__ == "__main__":
    flask_app.run(debug=True)
