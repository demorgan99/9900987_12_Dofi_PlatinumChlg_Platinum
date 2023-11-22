##Flask
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

## Proses mesin learning
import pickle, re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

##Swaagger desc
app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        "title": "API Documentation for Deep Learning",
        "description": "Dokumentasi API untuk for Deep Learning",
        "version": "1.0.0"
    },
    host = LazySring(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": '/flasgger_static',
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, config=swagger_config, template=swagger_template)

##Definisikan parameter untuk feature extraction
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, spilt=' ', lower=True)

##Definisikan label untuk sentimen
sentiment = ['negative', 'neutral', 'positive']

##Definisikan fungsi untuk cleansing
def cleansing(sent):
    string =  sent.lower()

    string = re.sub(r'[^a-bA-B0-9]',' ', string)
    return string
file = open("resource_of_cnn/x_pad_sequences.pickle", 'rb')
feature_file_from_cnn = pickle.load(file)
file.close()

model_file_from_cnn = load_model(model_file_from_cnn/model.h5)

file = open("resource_of_lstm/x_pad_sequences.pickle", 'rb')
feature_file_from_cnn = pickle.load(file)
file.close()

model_file_from_cnn = load_model(model_file_from_lstm/model.h5)

@swag_from('docs/cnn.yml', methods=['POST'])
@app.route('/predict_cnn', methods=['POST'])
deff cnn():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])
    prediction = model_file_from_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    
     json_response = {
        'status_code': 200,
        'description': "predict using CNN",
        'data' : {
        'text': original_text,
        'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from('docs/lstm.yml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
deff lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]
    
     json_response = {
        'status_code': 200,
        'description': "predict using LSTM",
        'data' : {
        'text': original_text,
        'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

@swag_from('docs/lstm.yml', methods=['POST'])
@app.route('/predict_lstm', methods=['POST'])

if __name__ == '__main__':
    app.run()