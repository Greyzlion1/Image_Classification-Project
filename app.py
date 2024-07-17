#Loading Libraries
from flask import Flask, request, redirect, url_for, render_template
from PIL import Image
import os
from flask import Flask, jsonify,request
from test import predict
app = Flask(__name__)
@app.route('/classify', methods=['POST'])
def classify():
    print(request.files)
    if 'image_to_predict' not in request.files:
        return jsonify(message="No file part"), 400
    file = request.files['image_to_predict']
    if file.filename == '':
        return jsonify(message="No selected file"), 400

    if file:
        file_path = f"images/{file.filename}"
        file.save(file_path)
        class_name = predict(file_path)
        return jsonify(message=class_name)

@app.route('/', methods=['GET'])
def upload_image():
    if request.method == 'POST':
        if 'Image' not in request.files:
            return 'No file part'
        file = request.files['Image']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Opening the uploaded image
            image = Image.open(file_path)
            image.show()
            return f'Image uploaded and opened successfully: {file_path}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)