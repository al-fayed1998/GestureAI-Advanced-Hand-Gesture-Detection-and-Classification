import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Vérification de la disponibilité du GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Configuration des générateurs d'images sans augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'C:/Users/ADMIN/Documents/IFI/Reconnaissance_des_formes/Projet/signcvz/Data_Augmentation1/train'
val_dir = 'C:/Users/ADMIN/Documents/IFI/Reconnaissance_des_formes/Projet/signcvz/Data_Augmentation1/val'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Charger le modèle VGG16 pré-entraîné
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Ajouter des couches personnalisées pour la classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Utilisez le nombre de classes dans vos données

# Créer le modèle final
model = Model(inputs=base_model.input, outputs=predictions)

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Définir le callback EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraîner le modèle avec Early Stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping]
)

# Évaluer le modèle
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Sauvegarder le modèle entraîné
model.save('hand_sign_detection_model_vgg16.h5')

# Sauvegarder les étiquettes des classes
class_indices = train_generator.class_indices
labels = dict((v, k) for k, v in class_indices.items())

with open('labels_vgg16.txt', 'w') as f:
    for i in range(len(labels)):
        f.write(f"{i} {labels[i]}\n")
