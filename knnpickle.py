import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime

# Record the start time
start_time = datetime.now()
print("Training Started At: "+str(start_time))

# Load data
df = pd.read_csv('user_act_big_prospect_id.csv')

# Convert CreatedOn to datetime
df['CreatedOn'] = pd.to_datetime(df['CreatedOn'])

# Group by ProspectID and concatenate event names to create event sequences
df['EventSequence'] = df.groupby('ProspectID')['EventName'].transform(lambda x: ' -> '.join(x))

# Drop duplicates to get unique sequences per user
unique_sequences = df[['ProspectID', 'EventSequence', 'Status']].drop_duplicates()


# Create a TfidfVectorizer to vectorize the event sequences
vectorizer = TfidfVectorizer(lowercase=False)
X = vectorizer.fit_transform(unique_sequences['EventSequence'])

# Train k-NN model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

# Save the model and vectorizer to a pickle file
with open('knn_model.pkl', 'wb') as f:
    pickle.dump((knn, vectorizer, unique_sequences), f)

print("Model and vectorizer saved to knn_model2.pkl")

# Record the end time
end_time = datetime.now()

# Calculate the time difference
time_difference = end_time - start_time


print(f"Start time: {start_time}")
print(f"End time: {end_time}")
print(f"Time taken: {time_difference}")
