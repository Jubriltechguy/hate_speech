from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)


@app.route('/')
def home():
	return render_template("home.html")

@app.route('/', methods = ['POST'])
def predict():
	#return render_template("result.html")
	

	df= pd.read_csv("data.csv")

	df_data = df[["class", "comments"]]
	df_x = df_data["comments"]
	df_y = df_data["class"]

	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.3, random_state=42)

	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression()
	clf.fit(X_train, y_train)
	clf.score(X_test, y_test)

	# #load the model
	# my_model = open("myFinalModel.pkl", "rb")
	# clf = pickle.load(my_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('home.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug = True)