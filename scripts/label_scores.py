"""
Example usage commands:

- for labelling the NEW dataset: python label_scores.py
- for labelling the initially used dataset: python label_scores.py --kaggle1
"""

import json
import pandas as pd
import os
import argparse

from labelling_scores.llm import LLM as LLM
from labelling_scores.image_to_url import local_image_to_data_url as local_image_to_data_url


def label_kaggle1(llm_client: LLM):
    """
    Labels the initially used Kaggle dataset with scores.
    Dataset linked at https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset?resource=download-directory

    args:
    - llm_client (LLM): the VLM / LLM client to be used for labelling
    """
    dataset_directory = "./Hands_kaggle1/Hands/"

    dataset_info = pd.read_csv("./Hands_kaggle1/HandInfo.csv")
    dataset_info

    previd = ""
    prevaspect = ""

    labels = []

    for index, row in dataset_info.iterrows():
        if "dorsal" in row['aspectOfHand']:
            continue
        
        if row['id'] == previd and row['aspectOfHand'] == prevaspect:
            continue

        previd = row['id']
        prevaspect = row['aspectOfHand']
        age = row['age']
        gender = row['gender']
        skincolor = row['skinColor']

        filename = row['imageName']

        user_prompt = f"""The following is an image of the user's palm.
    The user's information is as follows:
    Aspect of hand: {prevaspect}
    Age: {age}
    Gender: {gender}
    Skin color: {skincolor}

    Provide scores as you were instructed.
        """

        path = dataset_directory + filename
        imagedata = local_image_to_data_url(path)

        scores = llm_client.get_LLM_output(user_prompt=user_prompt, image_data=imagedata)

        try:
            scores_dict = json.loads(scores)
        except Exception as e:
            print(f"An error occurred: {e}")
            scores_dict = scores
        
        label = {
            "image": filename,
            "scores": json.dumps(scores_dict)
        }
        labels.append(label)

        labels_string = "" 
        for item in labels:
            item_json = json.dumps(item)
            labels_string = labels_string + item_json + "\n"

        with open("Hands_kaggle1/labels.json", "w") as f:
            f.write(labels_string)


def label_kaggle2(llm_client: LLM):
    """
    Labels the new Kaggle dataset with scores.
    Dataset linked at https://www.kaggle.com/datasets/feyiamujo/human-palm-images

    args:
    - llm_client (LLM): the VLM / LLM client to be used for labelling
    """
    # labelling for images inside subdirectory "FEMALE"
    all_files_female = []

    for root, _, files in os.walk("./Hands_kaggle2/FEMALE"):
        for file in files:
            full_file_path = os.path.join(root, file)
            all_files_female.append(full_file_path)

    labels_female = []

    for file in all_files_female:
        user_prompt = """The following is an image of the user's palm.
    The user's information is as follows:
    Gender: Female

    Provide scores as you were instructed.
        """
        path = file
        imagedata = local_image_to_data_url(path)

        scores = llm_client.get_LLM_output(user_prompt=user_prompt, image_data=imagedata)

        try:
            scores_dict = json.loads(scores)
        except Exception as e:
            print(f"An error occurred: {e}")
            scores_dict = scores

        label = {
            "image": file,
            "scores": json.dumps(scores_dict)
        }
        labels_female.append(label)

    labels_string_female = "" 
    for item in labels_female:
        item_json = json.dumps(item)
        labels_string_female = labels_string_female + item_json + "\n"

    with open("Hands_kaggle2/labels_female.json", "w") as f:
        f.write(labels_string_female)

    
    # labelling for images inside subdirectory "MALE"

    all_files_male = []

    for root, _, files in os.walk("./Hands_kaggle2/MALE"):
        for file in files:
            full_file_path = os.path.join(root, file)
            all_files_male.append(full_file_path)

    labels_male = []

    for file in all_files_male:
        user_prompt = """The following is an image of the user's palm.
    The user's information is as follows:
    Gender: Male

    Provide scores as you were instructed.
        """
        path = file
        imagedata = local_image_to_data_url(path)

        scores = llm_client.get_LLM_output(user_prompt=user_prompt, image_data=imagedata)

        try:
            scores_dict = json.loads(scores)
        except Exception as e:
            print(f"An error occurred: {e}")
            scores_dict = scores

        label = {
            "image": file,
            "scores": json.dumps(scores_dict)
        }
        labels_male.append(label)

    labels_string_male = "" 
    for item in labels_male:
        item_json = json.dumps(item)
        labels_string_male = labels_string_male + item_json + "\n"

    with open("Hands_kaggle2/labels_male.json", "w") as f:
        f.write(labels_string_male)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that uses a VLM to label a Kaggle palm images dataset with scores.")
    parser.add_argument(
        "--kaggle1",
        action="store_true",
        help="To label the initial dataset instead of the new one."
    )
    args = parser.parse_args()

    llm_client = LLM()

    if args.kaggle1:
        label_kaggle1(llm_client=llm_client)

    else:
        label_kaggle2(llm_client=llm_client)