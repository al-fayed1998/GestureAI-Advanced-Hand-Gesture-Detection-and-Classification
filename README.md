## GestureAI: Advanced Hand Gesture Detection and Classification

This project implements a real-time hand gesture detection and classification system using computer vision and deep learning. OpenCV is used for video capture and hand tracking, while a VGG16-based deep learning model classifies gestures. The system recognizes gestures such as “Headache,” “Help,” and “Dizziness,” with data augmentation techniques applied to enhance model performance.

### Key Features

	•	Real-time hand tracking via webcam.
	•	Gesture classification using a fine-tuned VGG16 model.
	•	Comprehensive data augmentation (rotation, zoom, flipping) to boost dataset variability.
	•	Model training with early stopping to mitigate overfitting.
	•	Enhanced optimization using the Adam optimizer for better accuracy.

### Installation

	1.	Clone the repository:

git clone https://github.com/al-fayed1998/GestureAI-Advanced-Hand-Gesture-Detection-and-Classification.git
cd GestureAI-Advanced-Hand-Gesture-Detection-and-Classification


	2.	Install required dependencies:

pip install -r requirements.txt


	3.	(Optional) Ensure a GPU-enabled setup for faster training.

### Dataset

The dataset comprises hand gestures divided into categories like “Headache,” “Help,” etc. You can either use an existing dataset or generate one by capturing images through the provided data collection script.

Data augmentation is employed to increase training sample diversity.

Usage

Train the model: Run the training script with your dataset.

python train.py

Run real-time gesture detection: After training, use the detection script for live gesture recognition.

python detect_gesture.py

### Model Architecture

	•	Base model: VGG16 (pre-trained on ImageNet, top layers removed).
	•	Custom layers: Includes a GlobalAveragePooling2D layer, a Dense layer with 512 units and ReLU activation, and a final Dense layer with softmax activation for classification.
	•	Optimizer: Adam optimizer with a 0.0001 learning rate.
	•	Loss function: Categorical crossentropy for multi-class classification.
	•	Early Stopping: Monitors validation loss to prevent overfitting.

Results

The model demonstrates high accuracy in detecting various hand gestures. Data augmentation and early stopping enhance generalization and reduce overfitting.

Contributions

Contributions are welcome! Feel free to fork the repository, submit pull requests, or raise issues with suggestions or bug reports.

License

This project is licensed under the MIT License.

Détection et Classification Avancée des Gestes de la Main

Ce projet propose un système de détection et de classification des gestes de la main en temps réel basé sur la vision par ordinateur et l’apprentissage profond. OpenCV est utilisé pour la capture vidéo et le suivi des mains, tandis qu’un modèle basé sur VGG16 effectue la classification. Le système reconnaît des gestes tels que « Mal de tête », « Aide », « Étourdissement », et plus encore, avec des techniques d’augmentation des données appliquées pour améliorer la performance du modèle.

Caractéristiques Principales

	•	Suivi des mains en temps réel via webcam.
	•	Classification des gestes avec un modèle VGG16 affiné.
	•	Augmentation des données (rotation, zoom, retournement) pour améliorer la diversité de l’ensemble d’entraînement.
	•	Entraînement avec arrêt précoce pour éviter le surajustement.
	•	Optimisation avec l’algorithme Adam pour une précision accrue.

Installation

	1.	Cloner le dépôt :

git clone https://github.com/yourusername/hand-gesture-detection.git
cd hand-gesture-detection


	2.	Installer les dépendances requises :

pip install -r requirements.txt


	3.	(Optionnel) Assurez-vous d’avoir un environnement compatible GPU pour un entraînement plus rapide.

Ensemble de Données

L’ensemble de données est composé de gestes de la main divisés en catégories telles que « Mal de tête », « Aide », etc. Vous pouvez utiliser un ensemble de données existant ou en générer un avec le script de collecte de données fourni.

L’augmentation des données est utilisée pour accroître la diversité des échantillons d’entraînement.

Utilisation

Entraînement du modèle : Lancez le script d’entraînement avec votre jeu de données.

python train.py

Détection en temps réel des gestes : Après l’entraînement, utilisez le script de détection pour la reconnaissance des gestes en direct.

python detect_gesture.py

Détails du Modèle

	•	Modèle de base : VGG16 (pré-entraîné sur ImageNet, couche supérieure enlevée).
	•	Couches personnalisées : Comprend une couche GlobalAveragePooling2D, une couche Dense avec 512 unités et activation ReLU, et une couche finale Dense avec activation softmax pour la classification.
	•	Optimiseur : Adam avec un taux d’apprentissage de 0,0001.
	•	Fonction de perte : Crossentropie catégorielle pour la classification multi-classes.
	•	Arrêt précoce : Surveille la perte de validation pour éviter le surajustement.

Résultats

Le modèle obtient une grande précision dans la reconnaissance des gestes de la main. L’augmentation des données et l’arrêt précoce contribuent à la généralisation et à la réduction du surajustement.

Contributions

Les contributions sont les bienvenues ! N’hésitez pas à forker le dépôt, soumettre des pull requests ou ouvrir des issues pour signaler des problèmes ou proposer des améliorations.

Licence

Ce projet est sous licence MIT.

Tu peux adapter et personnaliser les détails pour mieux répondre aux besoins de ton projet.
