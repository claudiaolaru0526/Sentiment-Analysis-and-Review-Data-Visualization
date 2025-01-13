import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob

# 1. Funcție pentru încărcarea recenziilor
def load_reviews_from_json(file_path):
    """
    Load reviews from a JSON file and return a Pandas DataFrame.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
        return pd.DataFrame()

# 2. Funcție pentru clasificarea sentimentelor pe baza polarității
def analyze_sentiment(comment):
    """
    Analyze the sentiment of a given comment using TextBlob.
    Classify:
    - Love: Polarity > 0.2
    - Hate: Polarity < -0.2
    - Neutral: -0.2 <= Polarity <= 0.2
    """
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    if polarity > 0.2:
        return "Love"
    elif polarity < -0.2:
        return "Hate"
    else:
        return "Neutral"

# 3. Funcție pentru calcularea similarității între recenzii
def calculate_cosine_similarity(reviews):
    """
    Calculate the cosine similarity matrix for a list of reviews.
    """
    if reviews.empty or len(reviews) == 0:
        print("No reviews available for similarity calculation.")
        return np.array([])  # Return an empty array if no data
    vectorizer = TfidfVectorizer()  # Convert text to numerical vectors
    tfidf_matrix = vectorizer.fit_transform(reviews)  # TF-IDF matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)  # Cosine similarity
    return similarity_matrix

# 4. Funcție pentru salvarea matricei de similaritate în CSV
def save_similarity_matrix_to_csv(matrix, output_path):
    """
    Save the cosine similarity matrix to a CSV file.
    """
    if matrix.size == 0:
        print("Similarity matrix is empty. Nothing to save.")
        return
    try:
        pd.DataFrame(matrix).to_csv(output_path, index=False)
        print(f"Similarity matrix saved to {output_path}")
    except Exception as e:
        print(f"Error while saving similarity matrix: {e}")

# 5. Funcția principală
def main():
    # Path to the JSON file containing reviews
    file_path = r"C:\Users\claud\Desktop\P3\reviews.json"
    output_csv_path = r"C:\Users\claud\Desktop\P3\similarity_matrix.csv"
    
    # Load the reviews into a DataFrame
    reviews = load_reviews_from_json(file_path)
    
    # Set pandas option to display all rows
    pd.set_option('display.max_rows', None)  # Show all rows of the DataFrame
    
    # Debugging steps to inspect the DataFrame
    print("Columns in DataFrame:", reviews.columns)  
    print("All reviews:")
    print(reviews)  # This will print all rows of the DataFrame

    # Check for required columns
    if "comment" not in reviews.columns or "rating" not in reviews.columns:
        raise ValueError("The 'comment' or 'rating' column is missing in the loaded data.")
    
    # Handle missing or empty comments and stars
    reviews = reviews.dropna(subset=["comment", "rating"])  # Drop rows where 'comment' or 'rating' is NaN
    reviews["comment"] = reviews["comment"].astype(str)  # Ensure comments are strings
    
    # Classify sentiments based on the polarity of the comment
    reviews["Sentiment"] = reviews["comment"].apply(analyze_sentiment)

    # Visualize sentiment distribution
    sentiment_counts = reviews["Sentiment"].value_counts()
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['#66b3ff', '#ff6666', '#99ff99'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()

    # Calculate cosine similarity between all comments
    similarity_matrix = calculate_cosine_similarity(reviews["comment"])
    
    # Display the results
    print("Updated DataFrame with Sentiment:")
    print(reviews)
    
    # Save cosine similarity matrix to a CSV file
    save_similarity_matrix_to_csv(similarity_matrix, output_csv_path)

    # Visualize the cosine similarity matrix (heatmap)
    if similarity_matrix.size > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='coolwarm', annot=False, fmt=".2f", square=True)
        plt.title("Cosine Similarity Heatmap")
        plt.xlabel("Reviews")
        plt.ylabel("Reviews")
        plt.show()

if __name__ == "__main__":
    main()
