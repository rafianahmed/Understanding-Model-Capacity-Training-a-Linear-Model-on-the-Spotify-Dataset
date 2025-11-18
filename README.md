Description:
This notebook explores the concept of model capacity by starting with the simplest neural architecture: a linear model. Using the Spotify dataset, the notebook trains a single-layer neural network and evaluates how well it performs on a regression task.

The goal is to demonstrate:

How a low-capacity model behaves during training

How underfitting appears in loss curves

Why model complexity matters in deep learning\

Spotify Regression – Linear Model (Underfitting Demo)

This project trains the simplest possible neural network — a linear model — on the Spotify dataset to explore concepts of model capacity, underfitting, and baseline performance in deep learning.

 Project Purpose

✔ Demonstrate how a single-layer model behaves on a regression task
✔ Build intuition on why deeper networks outperform linear models
✔ Provide a baseline to compare underfitting vs overfitting

This project is part of a learning module on Overfitting & Underfitting in Neural Networks.

 Model Architecture
Input → Dense(1) → Output


No hidden layers

No nonlinear activation

Equivalent to linear regression

Optimizer: Adam / SGD (depending on notebook version)

Loss: MAE (Mean Absolute Error)

This gives the model very low learning capacity, making it a perfect underfitting example.

 Dataset

The notebook uses the Spotify Tracks Dataset, containing audio statistics such as:

Danceability

Energy

Loudness

Acousticness

Tempo

Popularity (regression target)

 Dataset is NOT included in this repository
To run locally, download it from: (insert your dataset link or Kaggle source)

 Training

Example training call:

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[input_dim])
])

model.compile(
    optimizer='adam',
    loss='mae'
)

history = model.fit(X_train, y_train, epochs=200, batch_size=128)


Loss curves are plotted to analyze model behavior.

 Results

The linear model underfits the Spotify dataset

Validation and training loss converge at a high value

Demonstrates the need for hidden layers and nonlinearities

Later experiments (not included here) can compare:

Model	Capacity	Result
Linear	Low	 Underfits
Dense (3 layers)	Medium	✔ Good
Dense (10 layers)	High	⚠ May Overfit
🛠 Requirements
tensorflow
numpy
pandas
matplotlib
scikit-learn

 Running the Project
git clone <repo>
pip install -r requirements.txt
jupyter notebook overfitting_and_underfitting.ipynb

 Key Learning Outcomes

✔ Importance of model capacity
✔ How to detect underfitting from training curves
✔ Why deep learning models need non-linearity
