from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

vocab_size=15000
sentence_length=47

with open('tokenizer.pkl','rb') as f:

	tokens=pickle.load(f)

print("Vocabulary loaded")
news_predictor=load_model('news_predictor.h5')
print("Model loaded")

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':

		news_data=request.form['news_input']
		news_data=re.sub('[^a-zA-Z]',' ',news_data)
		news_data=news_data.lower()
		news_data=news_data.split()
		ps=PorterStemmer()
		news_data=[ps.stem(word) for word in news_data if word not in set(stopwords.words('english'))]
		news_data=' '.join(news_data)
		news_data=[news_data]
		token_news_data=tokens.texts_to_sequences(news_data)
		padded_news_data=pad_sequences(token_news_data,padding="post",maxlen=sentence_length)
		news_input=np.array(padded_news_data)
		news_result=news_predictor.predict(news_input)

		if news_result>0.5:

			result_value=1

		else:

			result_value=0

	return render_template('result.html',prediction=result_value)



if __name__ == '__main__':
	app.run(debug=True)

