# Medicinal Plant Detection App

An application for detecting medicinal plant species using a Convolutional Neural Network (CNN) based on the ResNet architecture. The app classifies over 200 medicinal plant species and provides an intuitive web interface for researchers and users.

## Project Structure

├── index.html # Frontend: HTML interface for user interaction ├── app.py # Backend: Flask application to serve the web interface ├── training.ipynb # Jupyter Notebook: Contains the model training code ├── model.h5 # Trained model weights (ResNet architecture) ├── requirements.txt # Python dependencies └── README.md # Documentation

markdown
Copy code

## Tech Stack
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Database:** MySQL
- **Machine Learning Framework:** TensorFlow

## Features
- Classification of 200+ medicinal plant species using a trained CNN model.
- Web interface for uploading plant images and viewing classification results.
- Streamlined data management with a MySQL database.
- User-friendly interface for researchers and professionals in the pharmaceutical industry.

## Prerequisites
1. Python 3.x installed on your machine.
2. Install dependencies:
pip install -r requirements.txt

markdown
Copy code
3. Set up a MySQL database:
- Create a database named `sih`.
- Import the database schema:
  ```
  mysql -u root -p sih < database.sql
  ```
- Update `db_config` in `app.py` if required.

## Usage

1. Run the Flask application:
python app.py

markdown
Copy code
2. Open your browser and navigate to `http://127.0.0.1:5000/`.
3. Upload an image of a medicinal plant to classify it.

## Training the Model
The model was trained using the `training.ipynb` file:
1. Open `training.ipynb` in Jupyter Notebook.
2. Ensure that TensorFlow and the required dependencies are installed.
3. Follow the notebook steps to train the model and save weights to `model.h5`.

## Deployment
This app can be deployed on any cloud platform supporting Flask applications, such as AWS EC2 or Heroku.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or contributions, please reach out via [GitHub](https://github.com/GoldSharon).
