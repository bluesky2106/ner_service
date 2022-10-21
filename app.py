import json

from flask import Flask, jsonify, request

from constants import *
from model import NERModel

app = Flask(__name__)
app.config["DEBUG"] = False

ner_model = NERModel()

@app.route('/', methods=['GET'])
def home():
    return "Homepage"

@app.route('/api/v1/ner', methods=['POST'])
def extract_entities():
	record = json.loads(request.data)
	model = record.get('model')
	text = record.get('text')

	if model is None or text is None:
		return jsonify({'error': 'wrong request body format'})
	if model not in AVAILABLE_MODELS:
		return jsonify({'error': 'only accept {} models'.format(AVAILABLE_MODELS)})
	
	response = []
	contents, labels = ner_model.predict_sentence(text, model)
	for content, label in zip(contents, labels):
		# print("{}\t{}".format(content, label))
		response.append({"content": content, "label": label})

	return jsonify(response)

if __name__ == '__main__':
  	app.run(host='0.0.0.0', port=8080)