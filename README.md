# 🚀 Object Detection Project

📌 Overview

This project focuses on Object Detection, utilizing a deep learning model to identify objects in images. The dataset was collected and processed using Python scripts, ensuring high-quality training data.

📷 Dataset

We have prepared a dataset that you can use directly:

🔗 Raw collected images: Download here

🔗 Organized and filtered images: Download here

The data collection process was handled using the provided Jupyter Notebook: data_collesion.ipynb.

📜 Data Collection & Preprocessing

The dataset was collected using a script that captures images and organizes them into labeled categories. After collection, the images were processed by:

Filtering out low-quality images

Resizing and normalizing images

Splitting data into training, validation, and test sets

The complete data preprocessing pipeline is detailed in data_collesion.ipynb.

🛠️ Training the Model

The model was trained using the following steps:

Loading and preprocessing the dataset

Building and compiling the object detection model

Training the model on labeled images

Evaluating model performance using validation data

You can find the full training pipeline in the Jupyter Notebook: object.ipynb.

The trained model file is available for download:

🔗 Trained model: Download here

📜 Code Explanation

🔹 data_collesion.ipynb

This notebook is responsible for collecting and organizing images for training. It includes:

Capturing images from a source (camera, dataset, etc.)

Labeling and saving images in an organized format

Preprocessing images (resizing, filtering, augmenting if necessary)

Exporting the dataset for model training

🔹 object.ipynb

This notebook contains the object detection model implementation. It covers:

Loading the dataset and applying necessary preprocessing

Defining a deep learning model using TensorFlow/Keras

Training the model with the prepared dataset

Evaluating and testing the model on unseen images

Saving the trained model for later use

🚀 How to Run

1️⃣ Clone the Repository

git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Required Libraries

To ensure smooth execution, install the following libraries:

pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn pillow tqdm

4️⃣ Run the Training Notebook

Open object.ipynb in Jupyter Notebook

Run all cells to train the model

📊 Results

The trained model successfully detects objects with high accuracy. Further optimizations can be applied for real-time inference.

🤝 Contributors

Noha Elblkasy - Project Lead

📞 Contact

For any inquiries or collaborations, feel free to reach out:
📧 nohaelblkasy@gmail.com🔗 LinkedIn
