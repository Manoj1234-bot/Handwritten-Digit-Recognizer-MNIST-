# 🔢 Handwritten Digit Recognizer

A deep learning web application that recognizes **handwritten digits (0–9)** using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. Users can draw a digit on an HTML5 canvas, and the Flask backend preprocesses the image before predicting the digit with confidence scores using a trained TensorFlow/Keras model.

## ✨ Features

* ✍️ Draw handwritten digits directly on the browser
* 🤖 CNN model trained on the MNIST dataset
* 🎯 Predicts digits from **0 to 9**
* 📊 Displays prediction confidence
* 📈 Shows probability for all digit classes
* ⚡ Fast real-time prediction using Flask
* 📱 Responsive and user-friendly interface
* 📉 Training accuracy and loss visualization

## 🛠️ Technologies Used

* Python 3
* TensorFlow / Keras
* Flask
* NumPy
* Pillow
* Matplotlib
* HTML5
* CSS3
* JavaScript

## 📸 Training Performance

The CNN model achieved approximately **98%+ accuracy** after training for 5 epochs. The training history shows increasing accuracy and decreasing loss, indicating effective learning.



```text
assets/training_history.png
```

## 📁 Project Structure

```text
handwritten-digit-recognizer/
│
├── app.py
├── Train_model.py
├── digit_model.keras
├── Requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── assets/
    └── training_history.png
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manoj1234-bot/Handwritten-Digit-Recognizer-MNIST-
```

### 2. Navigate to the project folder

```bash
cd handwritten-digit-recognizer
```

### 3. Install the required libraries

```bash
pip install -r Requirements.txt
```

### 4. Train the model

```bash
python Train_model.py
```

This downloads the MNIST dataset, trains the CNN model, and saves the trained model as:

```text
digit_model.keras
```

### 5. Run the Flask application

```bash
python app.py
```

### 6. Open in your browser

```text
http://127.0.0.1:5000
```

## ⚙️ How It Works

1. The CNN model is trained on the MNIST handwritten digit dataset.
2. Users draw a digit on the HTML5 canvas.
3. The drawing is sent to the Flask server as a Base64 image.
4. The image is converted to grayscale, inverted, resized to **28×28 pixels**, and normalized.
5. The trained CNN predicts the digit.
6. The application displays:

   * Predicted digit
   * Prediction confidence
   * Probability for all digits (0–9)

## 📊 Model Architecture

* Input Layer (28 × 28 × 1)
* Conv2D (32 Filters)
* MaxPooling2D
* Conv2D (64 Filters)
* MaxPooling2D
* Flatten Layer
* Dropout (0.5)
* Dense (64 Neurons)
* Output Layer (10 Softmax Classes)

## 📚 What I Learned

* Learned how to build and train a **Convolutional Neural Network (CNN)** using TensorFlow/Keras, preprocess image data, and classify handwritten digits from the MNIST dataset.
* Improved my understanding of deep learning, image preprocessing, Flask integration, model deployment, confidence prediction, and building AI-powered web applications.

## 🚀 Future Improvements

* 📱 Mobile-friendly drawing canvas
* 📷 Predict digits from uploaded images
* 📝 Recognize multiple handwritten digits
* 🤖 Improve accuracy using data augmentation
* ☁️ Deploy on Render or Railway
* 📊 Display prediction probability chart
* 🌙 Dark/Light mode support

## 👨‍💻 Author

**Manoj S**

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub. Contributions and feedback are always welcome!
