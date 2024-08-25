import nltk
import numpy as np
import torch
import json
import pickle
from nltk.stem import WordNetLemmatizer
import random
from train_model import ChatbotModel 

lemmatizer = WordNetLemmatizer()

# Load the trained model and data structures
words = pickle.load(open('chatbot/words.pkl', 'rb'))
classes = pickle.load(open('chatbot/classes.pkl', 'rb'))

input_size = len(words)
hidden_size = 128  # Must match the size used during training
model = ChatbotModel(input_size, hidden_size, len(classes))
model.load_state_dict(torch.load('chatbot/chatbot_model.pth', map_location=torch.device('cpu')))
model.eval()

with open('chatbot\intents.json') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow_tensor = torch.tensor(bow, dtype=torch.float32)
    with torch.no_grad():
        output = model(bow_tensor)
    output = torch.softmax(output, dim=0)
    ERROR_THRESHOLD = 0.25
    results = [[i, r.item()] for i, r in enumerate(output) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
