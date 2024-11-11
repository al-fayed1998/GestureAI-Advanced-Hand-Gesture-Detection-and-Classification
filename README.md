# Hand Gesture Detection and Classification Using Deep Learning

This project implements a real-time hand gesture detection and classification system using computer vision and deep learning. It uses OpenCV for video capture and hand tracking, and a VGG16-based deep learning model for gesture classification. The system can classify gestures such as "Headache," "Help," "Dizziness," and more, with data augmentation applied to improve model performance.

## Features
- Real-time hand detection using a webcam.
- Hand gesture classification using a fine-tuned VGG16 model.
- Data augmentation to enhance the dataset (rotation, zoom, flipping).
- Model training with early stopping to prevent overfitting.
- Optimized using the Adam optimizer for improved accuracy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-detection.git
   cd hand-gesture-detection
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt

3. Make sure you have a GPU-enabled setup (optional but recommended for faster training).


## Dataset
The dataset consists of hand gestures divided into categories like "Headache," "Help," and others. You can use your own dataset or generate one by capturing images from a webcam using the data collection script.

Data augmentation is applied to increase the diversity of training samples.

## Usage
Train the model: You can train the model using your dataset by running the training script.

```bash
python train.py
```
Run real-time gesture detection: Once the model is trained, you can run the real-time detection script using your webcam.

```bash
python detect_gesture.py
```

## Model Details
- Base model: VGG16 (pre-trained on ImageNet, with the top layer removed).
- Custom layers: A GlobalAveragePooling2D layer, followed by a Dense layer with 512 units and ReLU activation, and a final Dense layer for classification with softmax activation.
- Optimizer: Adam optimizer with a learning rate of 0.0001.
- Loss function: Categorical crossentropy for multi-class classification.
- Early Stopping: Applied to prevent overfitting by monitoring the validation loss.

## Results
The model achieves high accuracy in recognizing various hand gestures. The augmentation techniques and early stopping help generalize the model to new data while preventing overfitting.

## Contribution
Feel free to fork this repository, submit pull requests, or open issues if you encounter bugs or have suggestions for improvement.

## License
This project is licensed under the MIT License.

_________________________________________________________________________________________________________________________________________________________________________________________________________________

# Détection et classification des gestes de la main à l'aide de l'apprentissage profond

Ce projet met en œuvre un système de détection et de classification des gestes de la main en temps réel à l'aide de la vision par ordinateur et de l'apprentissage profond. Il utilise OpenCV pour la capture vidéo et le suivi des mains, et un modèle d'apprentissage profond basé sur VGG16 pour la classification des gestes. Le système peut classer des gestes tels que « mal de tête », « aide », « étourdissement », etc. L'augmentation des données est appliquée pour améliorer les performances du modèle.

## Caractéristiques
- Détection des mains en temps réel à l'aide d'une webcam.
- Classification des gestes de la main à l'aide d'un modèle VGG16 affiné.
- Augmentation des données pour améliorer l'ensemble des données (rotation, zoom, retournement).
- Apprentissage du modèle avec arrêt précoce pour éviter le surajustement.
- Optimisation à l'aide de l'optimiseur Adam pour une meilleure précision.

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/yourusername/hand-gesture-detection.git
   cd hand-gesture-detection
   ```
2. Installez les dépendances nécessaires :
   ```
   pip install -r requirements.txt

3. Assurez-vous d'avoir une configuration compatible avec le GPU (optionnel mais recommandé pour un entraînement plus rapide).


## Ensemble de données
L'ensemble de données se compose de gestes de la main divisés en catégories telles que « Mal de tête », « Aide » et autres. Vous pouvez utiliser votre propre ensemble de données ou en générer un en capturant des images à partir d'une webcam à l'aide du script de collecte de données.

L'augmentation des données est appliquée pour accroître la diversité des échantillons d'entraînement.

## Utilisation
Entraîner le modèle : Vous pouvez entraîner le modèle en utilisant votre jeu de données en exécutant le script d'entraînement.

```bash
python train.py
```
Exécuter la détection des gestes en temps réel : Une fois le modèle entraîné, vous pouvez exécuter le script de détection en temps réel en utilisant votre webcam.

``bash
python detect_gesture.py
``

## Détails du modèle
- Modèle de base : VGG16 (pré-entraîné sur ImageNet, avec la couche supérieure enlevée).
- Couches personnalisées : Une couche GlobalAveragePooling2D, suivie d'une couche Dense avec 512 unités et activation ReLU, et une couche Dense finale pour la classification avec activation softmax.
- Optimiseur : Optimiseur Adam avec un taux d'apprentissage de 0,0001.
- Fonction de perte : Crossentropie catégorielle pour la classification multi-classes.
- Arrêt précoce : Appliqué pour éviter le surajustement en surveillant la perte de validation.

## Résultats
Le modèle atteint une grande précision dans la reconnaissance des différents gestes de la main. Les techniques d'augmentation et l'arrêt précoce permettent de généraliser le modèle à de nouvelles données tout en évitant l'ajustement excessif.

## Contribution
N'hésitez pas à forker ce dépôt, à soumettre des requêtes ou à ouvrir des problèmes si vous rencontrez des bogues ou si vous avez des suggestions d'amélioration.

## Licence
Ce projet est placé sous licence MIT.
