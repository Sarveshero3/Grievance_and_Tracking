import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle

# Ensure the NLTK data is downloaded
import nltk
nltk.data.path.append('C:/Users/Sarvesh/AppData/Roaming/nltk_data')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load the intents file
with open('chatbot\intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create our training data
training = []
output_empty = [0] * len(classes)

# Training set, bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists. X - patterns, Y - intents
train_x = np.array(list(training[:, 0]), dtype=np.float32)
train_y = np.array(list(training[:, 1]), dtype=np.float32)

# Convert to PyTorch tensors
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)

class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model
input_size = len(train_x[0])
hidden_size = 128
output_size = len(classes)

model = ChatbotModel(input_size, hidden_size, output_size)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
batch_size = 5
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_x), batch_size):
        x_batch = train_x[i:i+batch_size]
        y_batch = train_y[i:i+batch_size]
        
        # Forward pass
        outputs = model(x_batch)
        loss = model.loss_fn(outputs, torch.max(y_batch, 1)[1])
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

import os

os.makedirs('chatbot', exist_ok=True)

# Save the model
torch.save(model.state_dict(), os.path.join('chatbot', 'chatbot_model.pth'))

# Save the data structures
with open(os.path.join('chatbot', 'words.pkl'), 'wb') as f:
    pickle.dump(words, f)

with open(os.path.join('chatbot', 'classes.pkl'), 'wb') as f:
    pickle.dump(classes, f)

print("Model training complete and saved!")
# Save the model
torch.save(model.state_dict(), 'chatbot/chatbot_model.pth')

# Save the data structures
pickle.dump(words, open('chatbot/words.pkl', 'wb'))
pickle.dump(classes, open('chatbot/classes.pkl', 'wb'))

print("Model training complete and saved!")
