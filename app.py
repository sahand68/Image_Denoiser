#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import cv2
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

#decoding an image from base64 into raw representation
def convertImage(imgData1):
	cv2.imwrite('output.png',imgData1)
	#print(imgstr)
	

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	#whenever the predict method is called, we're going
	#to input the user drawn character as an image into the model
	#perform inference, and return the classification
	#get the raw data format of the image
	imgData = request.get_data()
	#encode it into a suitable format
	convertImage(imgData)
	
	#read the image into memory
	x = cv2.imread('output.png',0)
	x=cv2.resize(x,(0,0),fx = 0.25, fy = 0.25)
	x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	x = x.astype('float32')/255
	newX,newY = x.shape[0], x.shape[1]  
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.reshape(1,newX,newY, 1)
	x = np.clip(x, 0.,1.)
	#make it the right size
	#imshow(x)
	#convert to a 4D tensor to feed into our model

	
	#in our computation graph
	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		out = np.concatonate(out)
		out= np.vstack([np.hstack(i) for i in out])
		plt.imshow(x, cmap= 'gray')
		
		return 
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
