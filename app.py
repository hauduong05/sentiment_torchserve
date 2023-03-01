from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

with open('index_to_name.json', 'r') as f:
    class_names = json.loads(f.read())

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello'

@app.route('/predictions/sentiment', methods=['POST'])
def predict():
    data = request.json
    text = data.get('input')
    tokenized_data = tokenizer.encode_plus(
                    text,
                    max_length=50,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )

    outputs = model(**tokenized_data)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = torch.argmax(probabilities, axis=1)
    predictions = predictions.tolist()
    predictions = [class_names[str(label)] for label in predictions]
    out = {i: predictions[i] for i in range(len(predictions))}
    return out


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)


