# feature_extraction.py

from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 'Simple Ratio', 'partial Ratio', 'Token Set Ratio', 'word match percentage', 'last word match', 'character_matching_percentage', 'ngrams'


#Function to calculate simple ratio
def calculate_simple_ratio(string1, string2):
    return fuzz.ratio(string1, string2)


#Function to calculate partial ratio
def calculate_partial_ratio(string1, string2):
    x1= fuzz.partial_ratio(string1, string2)
    x2= fuzz.partial_ratio(string1, string2)
    return max(x1,x2)


# Function to calculate token sort ratio
def calculate_token_sort_ratio(string1, string2):
    return fuzz.token_sort_ratio(string1, string2)


# Function to calculate token set ratio
def calculate_token_set_ratio(string1, string2):
    return fuzz.token_set_ratio(string1, string2)


# Function to calculate word matching percentage
def calculate_word_matching_percentage(string1, string2):
    # Split strings into words
    # print("strings", string1, " ", string2)
    words1 = set(string1.split())
    words2 = set(string2.split())

    # Calculate the intersection of words
    common_words = words1.intersection(words2)

    # Calculate the percentage of matching words
    matching_percentage = (len(common_words) / len(words1.union(words2))) * 100

    return matching_percentage


# Function to calculate first word matching percentage
def first_word_match(string1, string2):
    # Split strings into words
    words1 = string1.split()
    words2 = string2.split()

    # Check if there are at least two words in each string
    if len(words1) >= 1 and len(words2) >= 1:
        # Compare the first words
        return words1[0] == words2[0]
    else:
        # If any of the strings has fewer than two words, return False
        return False
    

# Function to calculate last word matching percentage
def last_word_match(string1, string2):
    # Split strings into words
    words1 = string1.split()
    words2 = string2.split()

    # Check if there are at least two words in each string
    if len(words1) >= 1 and len(words2) >= 1:
        # Compare the last words
        return words1[-1] == words2[-1]
    else:
        # If any of the strings has fewer than two words, return False
        return False


# Function to calculate charater matching percentage
def character_matching_percentage(string1, string2):
    # Calculate the length of the longer string
    max_length = max(len(string1), len(string2))

    # Calculate the number of matching characters
    matching_characters = sum(c1 == c2 for c1, c2 in zip(string1, string2))

    # Calculate the percentage of matching characters
    matching_percentage = (matching_characters / max_length) * 100

    return matching_percentage


# Function to calculate cosine similarity
def calculate_cosine_similarity(string1, string2):
    # Combine preprocessed strings for vectorization
    sentence1 = " ".join(string1.split())
    sentence2 = " ".join(string2.split())

    # Create vectors
    vectorizer = CountVectorizer().fit_transform([sentence1, sentence2])
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))

    return cosine_sim[0, 0]


def get_ngrams(text, n):
    """
    Generate character n-grams for a given text.
    """
    ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return ngrams

def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union != 0 else 0

def calculate_ngrams_similarity(string1, string2):
    ngrams1 = set(get_ngrams(string1, 2))
    ngrams2 = set(get_ngrams(string2, 2))

    # Calculate Jaccard similarity
    similarity = jaccard_similarity(ngrams1, ngrams2)
    
    return similarity
    

def common_prefix_percentage(string1, string2):
    """
    Calculate the percentage of common prefix characters out of the total characters.
    """
    min_len = min(len(string1), len(string2))
    
    # Calculate the common prefix length
    common_prefix_len = 0
    for i in range(min_len):
        if string1[i] == string2[i]:
            common_prefix_len += 1
        else:
            break

    # Calculate the percentage
    if min_len == 0:
        return 0.0  # Avoid division by zero
    else:
        percentage = (common_prefix_len / min_len) * 100
        return percentage


def common_suffix_percentage(string1, string2):
    """
    Calculate the percentage of common prefix characters out of the total characters.
    """
    min_len = min(len(string1), len(string2))
    
    common_suffix_len = 0
    for i in range(1, min_len + 1):
        if string1[-i] == string2[-i]:
            common_suffix_len += 1
        else:
            break

    # Calculate the percentage
    if min_len == 0:
        return 0.0  # Avoid division by zero
    else:
        percentage = (common_suffix_len / min_len) * 100
        return percentage
    

def common_prefix_suffix_percentage(string1, string2):
    # Calculate the length of the common prefix
    prefix_len = 0
    min_len = min(len(string1), len(string2))
    for i in range(min_len):
        if string1[i] == string2[i]:
            prefix_len += 1
        else:
            break

    # Calculate the length of the common suffix
    suffix_len = 0
    for i in range(1, min_len + 1):
        if string1[-i] == string2[-i]:
            suffix_len += 1
        else:
            break

    # Calculate the percentage of common prefix and suffix
    total_len = len(string1) + len(string2) - prefix_len - suffix_len
    percentage = (prefix_len + suffix_len) / total_len * 100 if total_len > 0 else 0

    return percentage
