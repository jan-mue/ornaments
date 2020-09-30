import os
import pandas as pd
import random
import shutil

from .download_images import load_ornaments_df


def filter_data(ornaments, base_dir="../../data/"):
    result = []
    for index, row in ornaments.iterrows():
        pre, ext = os.path.splitext(str(row["Filename"]))
        filename = pre + ".png"

        if not os.path.exists(base_dir + "images/" + filename):
            continue

        if row["RosetteFlag"] == "YES" or row["Field13"] == "YES":
            continue

        result.append((filename, row["Group"]))
    return result


def move_images(image_labels, postfix, base_dir="../../data/"):
    target_dir = base_dir + "images_" + postfix
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    pd.DataFrame(image_labels, columns=["filename", "group"]).to_csv(base_dir + postfix + ".csv", index=False)

    for filename, label in image_labels:
        shutil.copy("images/" + filename, target_dir)


if __name__ == "__main__":

    ornaments = load_ornaments_df()
    ornaments = filter_data(ornaments)

    n = len(ornaments)
    random.shuffle(ornaments)
    train_split = ornaments[:int(0.8 * n)]
    test_split = ornaments[int(0.8 * n):]

    move_images(train_split, "train")
    move_images(test_split, "test")
