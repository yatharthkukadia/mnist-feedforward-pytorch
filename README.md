# MNIST Feedforward Neural Network using PyTorch

This project implements a simple **feedforward neural network** for handwritten digit classification using the **MNIST dataset** in PyTorch.

## 🚀 Features

- Uses `torchvision` to download and transform the MNIST dataset
- Implements a custom `FeedForwardNet` class using `nn.Module`
- Trains using `CrossEntropyLoss` and `Adam` optimizer
- Supports CPU and GPU training
- Saves the trained model to a `.pth` file

## 📦 Requirements

Make sure you have Python and the required libraries installed:

```bash
pip install torch torchvision
#   m n i s t - f e e d f o r w a r d - p y t o r c h 

🧠 Model Architecture
Input Layer: 784 nodes (28x28 images)

Hidden Layer: 256 neurons with ReLU activation

Output Layer: 10 neurons with Softmax activation

📁 Files
main.py: The full training script

feedforwardnet.pth: The saved trained model

README.md: Project documentation

How to Run

python main.py
 
The model will be trained for 10 epochs and saved as feedforwardnet.pth.

💾 Saving the Model
The trained model is saved using:

torch.save(model.state_dict(), 'feedforwardnet.pth') 

📊 Sample Output

Epoch 1
Loss: 0.4321
-----------------------------
...
Training is Done.
Model trained and stored at feedforwardnet.pth
Using cuda device (I am using cuda for you it may be cpu)

📌 Notes
This is a simple starter example for learning PyTorch.

You can experiment by modifying the architecture or using different optimizers/loss functions.

🧑‍💻 Author
Yatharth Kukadia
Student of Artificial Intelligence & Data Science
