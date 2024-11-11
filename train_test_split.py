import os
import shutil
from sklearn.model_selection import train_test_split

# Chemin du dossier contenant les images de toutes les catégories
dataset_dir = 'C:/Users/ADMIN/Documents/IFI/Reconnaissance_des_formes/Projet/signcvz/Data_Augmentation1'  # Remplacez par le chemin de votre dataset

# Dossiers de sortie pour les ensembles d'entraînement et de validation
train_dir = 'C:/Users/ADMIN/Documents/IFI/Reconnaissance_des_formes/Projet/signcvz/Data_Augmentation1/train'
val_dir = 'C:/Users/ADMIN/Documents/IFI/Reconnaissance_des_formes/Projet/signcvz/Data_Augmentation1/val'

# Créer les dossiers d'entraînement et de validation s'ils n'existent pas
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print(f"Création du dossier d'entraînement : {train_dir}")
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    print(f"Création du dossier de validation : {val_dir}")

# Diviser les images de chaque catégorie en ensembles d'entraînement et de validation
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if os.path.isdir(category_path) and category not in ['train', 'val']:  # Vérifier que c'est un dossier et exclure 'train' et 'val'
        images = os.listdir(category_path)
        
        # Vérifier s'il y a des images dans le dossier
        if len(images) == 0:
            print(f"Aucune image trouvée dans le dossier {category_path}.")
            continue
        
        # Diviser les images en 80% pour l'entraînement et 20% pour la validation
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        
        # Créer les sous-dossiers pour chaque catégorie dans les dossiers d'entraînement et de validation
        train_category_path = os.path.join(train_dir, category)
        val_category_path = os.path.join(val_dir, category)
        if not os.path.exists(train_category_path):
            os.makedirs(train_category_path)
            print(f"Création du sous-dossier d'entraînement pour {category} : {train_category_path}")
        if not os.path.exists(val_category_path):
            os.makedirs(val_category_path)
            print(f"Création du sous-dossier de validation pour {category} : {val_category_path}")
        
        # Copier les images vers les dossiers d'entraînement
        for img in train_images:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(train_category_path, img)
            shutil.copy(src_path, dst_path)
        
        # Copier les images vers les dossiers de validation
        for img in val_images:
            src_path = os.path.join(category_path, img)
            dst_path = os.path.join(val_category_path, img)
            shutil.copy(src_path, dst_path)

print("Les images ont été divisées et copiées avec succès.")
