import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurer le générateur d'images avec data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Chemins des dossiers d'entrée et de sortie
input_dir = 'Data_pref\Maux_tete_pref'  # Remplacez par le chemin de votre dossier d'entrée
output_dir = 'Data_Augmentation\Maux_tete_augm'  # Remplacez par le chemin de votre dossier de sortie

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configuration du flux de données depuis le dossier d'entrée
batch_size = 1
image_size = (150, 150)  # Modifiez selon la taille de vos images

generator = datagen.flow_from_directory(
    input_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode=None,  # Nous ne nous soucions pas des labels dans ce cas
    save_to_dir=output_dir,
    save_prefix='aug',
    save_format='jpg'
)

# Nombre d'images à générer par image source
num_augmented_images_per_image = 5

# Vérifier le nombre de fichiers trouvés
if len(generator.filepaths) == 0:
    print(f"Aucune image trouvée dans le dossier {input_dir}. Vérifiez la structure du dossier et les extensions des fichiers.")
else:
    # Générer et sauvegarder les images augmentées
    num_images_generated = 0
    for i in range(len(generator.filepaths) * num_augmented_images_per_image):
        generator.next()
        num_images_generated += 1

    print(f"{num_images_generated} images augmentées ont été sauvegardées dans le dossier {output_dir}")
