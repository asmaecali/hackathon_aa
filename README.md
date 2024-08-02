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

Traitement et Prédiction des Vidéos avec YOLO et AutoGluon

Ce script traite des vidéos pour extraire les coordonnées des points clés humains à l'aide du modèle YOLOv8, puis utilise AutoGluon pour prédire les classes d'action basées sur ces données.
Prérequis

    Python 3.x
    ultralytics, pandas, numpy, autogluon

Installez les dépendances avec :

bash

pip install ultralytics pandas numpy autogluon

Fonctionnalités

    Extraction des Points Clés :
        Utilise YOLOv8 pour détecter les points clés dans les vidéos.
        Normalise les coordonnées des points clés et sauvegarde les données dans un fichier CSV.

    Prédiction des Classes :
        Charge un modèle AutoGluon pré-entraîné pour prédire les classes d'action basées sur les données extraites.
        Sauvegarde les résultats de la prédiction dans un autre fichier CSV.

Usage

    Modifiez le chemin dans ar(path) pour pointer vers votre dossier contenant les vidéos.

    Exécutez le script :

    bash

    python votre_script.py

Exemple

python

ar("/chemin/vers/votre/dossier/videos")

Résultats

    Les coordonnées des points clés et les classes originales sont sauvegardées dans kp_test_one_line2.csv.
    Les prédictions sont sauvegardées dans kp_inf_one_line2.csv.

    Argument : video_path : Chemin de la vidéo à analyser.

Utilisation

Modifiez les chemins dans le code puis exécutez 

