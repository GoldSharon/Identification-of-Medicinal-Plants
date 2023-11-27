from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import mysql.connector
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("medicinal_plant_model.h5")

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "gold",
    "database": "sih"
}

# Define a function for preprocessing the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict the plant from an input image
def predict_plant(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(img)

    # Define the list of plant names
    query = "SELECT Scientific_Name FROM sihdata"
    
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    cursor.execute(query)
    
    plant_names = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    
    # Get the top predicted class
    top_prediction = np.argmax(predictions[0])

    # Map the prediction index to the corresponding plant name
    predicted_plant = plant_names[top_prediction]

    return predicted_plant

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Save the uploaded image temporarily
            uploaded_image_path = "temp_image.jpg"
            file.save(uploaded_image_path)
            
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            # Predict the plant from the uploaded image
            predicted_plant = predict_plant(uploaded_image_path)
            
            query = "SELECT Genus_Name, Species_Name, Usable_Part, Uses, Places FROM sihdata WHERE Scientific_Name = %s"
            
            cursor.execute(query, (predicted_plant,))
            plant_info = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            # Delete the temporary image
            os.remove(uploaded_image_path)
           
            # Return the predictions as JSON response
            response_data = {
                'Scientific_Name': predicted_plant,
                'genusName': plant_info[0],
                'speciesName': plant_info[1],
                'whichPart': plant_info[2],
                'usesOfPart': plant_info[3],
                'findPlace': plant_info[4]
            }

            return jsonify(response_data)
    return render_template('Final.html')

if __name__ == '__main__':
    app.run(debug=True)
