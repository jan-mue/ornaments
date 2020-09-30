import pandas as pd
import requests
import time
import os


def get_filename(row):
    if row["Field12"] == "YES":
        if row["Field13"] == "YES":
            return "owe_2/upload/"+row["Filename"]+"_iOrnamentPictureSp.png"
        else:
            return "owe_2/upload/"+row["Filename"]+"_iOrnamentPicture.png"

    result = "owe/PlaneOrnaments/"

    if row["RosetteFlag"] == "YES":
        result = "owe/Rosettes/"

    return result + row["Filename"]


def load_ornaments_df(base_dir="../../"):
    df1 = pd.read_csv(base_dir+"ornaments.csv", header=0, quotechar='"')
    df2 = pd.read_csv(base_dir+"ornaments1.csv", header=0, quotechar='"')

    df1["Field13"] = "NO"

    df = pd.concat([df1, df2])
    return df.drop_duplicates(["Filename", "Field13"])


if __name__ == "__main__":
    ornaments = load_ornaments_df()

    if not os.path.exists("images"):
        os.makedirs("images")

    for index, row in ornaments.iterrows():
        fn = get_filename(row)
        r = requests.get("http://science-to-touch.com/en/" + fn)

        time.sleep(0.5)

        if r.status_code == 404:
            continue

        print("Downloaded file " + fn)

        pre, ext = os.path.splitext(row["Filename"])

        with open("images/" + pre + ".png", "wb") as f:
            f.write(r.content)
