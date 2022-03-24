from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

classes = ['female','male']

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    face_crop = image.load_img(img_path, target_size=(218,178))
    face_crop = image.img_to_array(face_crop)/255.0
    face_crop = np.expand_dims(face_crop, axis=0)

    conf = model.predict(face_crop)[0]
    
    idx = np.argmax(conf)
    label = classes[idx]
    return (label,conf[idx])


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)
		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	app.run(debug = True)