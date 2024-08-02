Prédiction d'Actions Vidéo avec ConvLSTM

Ce script prédit des actions à partir de séquences vidéo en utilisant un modèle ConvLSTM.
Prérequis

    Python 3.x
    Bibliothèques : OpenCV, NumPy, TensorFlow

Installez les dépendances avec :

bash

pip install opencv-python numpy tensorflow

Variables Clés

    CLASSES_LIST : Classes de prédiction (["normal", "anormal"]).
    MODEL_PATH : Chemin vers le modèle ConvLSTM pré-entraîné.

Fonctions
frames_extraction(video_path)

Extrait des frames d'une vidéo.

    Argument : video_path : Chemin de la vidéo.
    Retour : Liste de frames normalisées.

predict_single_action(video_file_path, model_file_path)

Prédit l'action d'une vidéo en utilisant le modèle ConvLSTM.

    Arguments :
        video_file_path : Chemin de la vidéo.
        model_file_path : Chemin du modèle.

ar(video_path)

Fonction d'exemple pour prédire l'action d'une vidéo.

    Argument : video_path : Chemin de la vidéo à analyser.

Utilisation

Modifiez les chemins dans le code puis exécutez 

