# Imports/Depedancies
from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Info on oranges
orange_info = {
  "Tangerine Species":"Tangerines, a smaller and less rounded type of orange, is known for its sweetness. The peel of tangerines are much thinner, making them a great snack to have whenever. Tangerines are mostly eaten by hand, although they are used in many dishes as well.",
  "Valencia Orange":"The Valencia Orange, a sweet orange cultivar, is named after the famed oranges of Valencia, Spain. This type of orange is mainly cultivated for processing or juicing. However, its excellent taste and color make it great for fresh fruit markets as well.",
  "Tangelo Orange":'The Tangelo Orange, which is very juicy, has a tart and tangy taste. Its peel is looser, making it easier to peel than typical oranges. The Tangelo Orange is known for the "nipple" at its stem.',
  "Blood Orange":"The Blood Orange, also known as the Raspberry Orange, is famous for its crimson, blood-colored flesh. The skin on these oranges are typically thicker and harder to peel. Blood oranges have a more tart and floral taste compared to other types of oranges.",
  "Navel Orange":"The Navel Orange is the most general and typical orange. This orange has many varieties that are commonly cultivated and eaten today. Navel oranges are used in many dishes such as salads, deserts, and sauces.",
  "Not An Orange":"You don't have an orange... I don't know what else to tell you"
}

# Unfinished
orange_file = {
  "Tangerine Species":"Tangerines.png",
  "Valencia Orange":"Valencia.jpg",
  "Tangelo Orange":"Tangelo Oranges.png",
  "Blood Orange":"Blood.jpg",
  "Navel Orange":"Navel.jpg",
  "Not An Orange":"question.png"
}

# Init app
app = Flask('app', static_folder='src')

# Index/Input
@app.route('/')
def index():
  audio_files = ["cocomall.mp3"]
  return render_template('main.html', audio_files=audio_files)

# Output page
@app.route('/output', methods=['POST'])
def upload():
  if request.method == 'POST':
    file = request.files['image']
    # Save the file to the server
    file.save('src/uploads/' + file.filename)
    # Load model
    model = load_model(r".\src\uploads\model\keras_model.h5", compile=False)
    class_names = open(r".\src\uploads\model\labels.txt", "r").readlines()
    # Init model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Open image/Get input
    image = Image.open(fr".\\src\\uploads\\{file.filename}").convert("RGB") # change file directory to your machine/enviorment
    # Computation/AI stuff
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    # Get name of predicted
    class_name = class_names[index]
    # Get confidence of prediction
    confidence_score = prediction[0][index]
    # Parse Class
    Class = class_name[2:].replace("\n", "")
    # Pass data to upload.html and render
    return render_template('upload.html', Class=Class, confidence=round(confidence_score*100), fact=orange_info[Class], filename=file.filename, orange_picture=orange_file[Class])

# Handle 404
@app.errorhandler(404)
def page_not_found(error):
  return render_template('404.html'), 404

# Host
app.run(host="0.0.0.0", port=8080, debug=True)