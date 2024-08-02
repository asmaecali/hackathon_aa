from ultralytics import YOLO
import pandas as pd
import os
import numpy as np

import pandas as pd
from autogluon.tabular import TabularPredictor



def process_videos_in_folders(base_folder):
    model = YOLO("yolov8x-pose.pt")
    keypoints_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    all_data = []
    for class_folder_name in os.listdir(base_folder):
        class_folder_path = os.path.join(base_folder, class_folder_name)

        if os.path.isdir(class_folder_path):  
            video_class = class_folder_name  

            for video_file_name in os.listdir(class_folder_path):
                video_file_path = os.path.join(class_folder_path, video_file_name)
                if os.path.isfile(video_file_path) and (video_file_name.endswith(".mp4") or video_file_name.endswith(".avi")):
                    try:
                        results = model.track(video_file_path, stream=True)
                    except Exception as e:
                        print(f"Error processing {video_file_name}: {e}")
                        continue

                    video_data = {'class': video_class}
                    frames = []
                    frame_number = 0

                    for r in results:
                        if not hasattr(r, 'keypoints') or r.keypoints is None or len(r.keypoints.xy.cpu().numpy()) == 0:
                            frame_data = {f"{name}_{frame_number}_x": pd.NA for name in keypoints_names}
                            frame_data.update({f"{name}_{frame_number}_y": pd.NA for name in keypoints_names})
                        else:
                            keypoints = r.keypoints.xy.cpu().numpy()
                            if keypoints.size == 0:
                                frame_data = {f"{name}_x_{frame_number}": pd.NA for name in keypoints_names}
                                frame_data.update({f"{name}_y_{frame_number}": pd.NA for name in keypoints_names})
                            else:
                                keypoints = keypoints[0]
                                frame_width = r.orig_shape[1]
                                frame_height = r.orig_shape[0]

                                frame_data = {}
                                for i, name in enumerate(keypoints_names):
                                    x = keypoints[i][0]
                                    y = keypoints[i][1]
                                    x_normalized = x / frame_width if x != 0 else pd.NA
                                    y_normalized = y / frame_height if y != 0 else pd.NA
                                    frame_data[f"{name}_{frame_number}_x"] = x_normalized
                                    frame_data[f"{name}_{frame_number}_y"] = y_normalized

                        frames.append(frame_data)
                        frame_number += 1

                    if len(frames) > 48:
                        frames = frames[:48]
                    elif len(frames) < 48:
                        last_frame = frames[-1] if frames else {}
                        while len(frames) < 48:
                            frames.append(last_frame)

                    for frame_number in range(48):
                        frame_data = frames[frame_number] if frame_number < len(frames) else {}
                        for name in keypoints_names:
                            video_data[f"{name}_{frame_number}_x"] = frame_data.get(f"{name}_{frame_number}_x", pd.NA)
                            video_data[f"{name}_{frame_number}_y"] = frame_data.get(f"{name}_{frame_number}_y", pd.NA)

                    all_data.append(video_data)
                    print(f"Processed {video_file_name} in class {video_class}")

    if all_data:
        csv_path = os.path.join(base_folder, "kp_test_one_line2.csv")
        final_df = pd.DataFrame(all_data)
        final_df.to_csv(csv_path, index=False)
        print(f"All keypoints coordinates and class information saved to {csv_path}")
    else:
        print(f"No video data processed in {base_folder}")

def ar(path: str):
    process_videos_in_folders(path)

    model_path = "/home/c3po/AR/pose/scripts/AutogluonModels/ag-20240729_144955"
    loaded_model = TabularPredictor.load(model_path)

    inference_csv_path = os.path.join(path, "kp_test_one_line2.csv")
    inference_data = pd.read_csv(inference_csv_path)
    y_inference_pred = loaded_model.predict(inference_data)
    inference_data['predicted_class'] = y_inference_pred
    columns = list(inference_data.columns)
    if 'class' in columns:
        columns.insert(1, columns.pop(columns.index('predicted_class')))
    inference_data = inference_data[columns]

    output_path = os.path.join(path, "kp_inf_one_line2.csv")

    inference_data.to_csv(output_path, index=False)

    print("Prédictions et classes originales :")
    print(inference_data[['class', 'predicted_class']])
    print(f"Les résultats ont été sauvegardés dans {output_path}")

ar("/home/c3po/AR/pose/2_classes/test")