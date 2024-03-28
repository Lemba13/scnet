"""Script to convert soccernet annotations to yolo format"""
import configparser
import datetime
import glob
import os
import shutil
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import pytz
from tqdm import tqdm

ist = pytz.timezone("Asia/Kolkata")


def parse_gameinfo(file_path: str) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    Parse tracklets from an INI file and return them as a dictionary.

    Args:
        file_path (str): The path to the INI file.

    Returns:
        Tuple[Dict[int, str], Dict[int, int]]: A tuple containing a dictionary
        with tracklet IDs as keys and tracklet values as values, and a dictionary
        mapping tracklet IDs to categories.
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    num_tracklets = int(config["Sequence"]["num_tracklets"])
    tracklets = {}
    id_map = {}
    for i in range(1, num_tracklets + 1):
        key = f"trackletID_{i}"
        value = config["Sequence"][key]
        if "goalkeeper" in value:
            id_map[i] = 1
        elif "player" in value:
            id_map[i] = 2
        elif "referee" in value:
            id_map[i] = 3
        else:
            id_map[i] = 0
        tracklets[i] = value

    return tracklets, id_map


def parse_seqinfo(file_path: str) -> Dict[str, str]:
    """
    Parse tracklets from an INI file and return them as a dictionary.

    Args:
        file_path (str): The path to the INI file.

    Returns:
        Dict[str, str]: A dictionary containing tracklet IDs as keys and tracklet values as values.
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    key_value_pairs = config["Sequence"]
    parsed_data = dict(key_value_pairs)
    return parsed_data


def rectangle_to_polygon_normalized(
    x0: int, y0: int, width: int, height: int, img_width: int, img_height: int
) -> List:
    """
    Convert rectangle coordinates to normalized polygon coordinates representing the four corners of the rectangle.

    Args:
        x0 (int): The x-coordinate of the top-left corner of the rectangle.
        y0 (int): The y-coordinate of the top-left corner of the rectangle.
        width (int): The width of the rectangle.
        height (int): The height of the rectangle.
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        List[Tuple[float, float]]: A list of tuples representing the normalized polygon coordinates of the rectangle's four corners.
    """
    x1, y1 = x0 + width, y0
    x2, y2 = x1, y1 + height
    x3, y3 = x0, y0 + height

    x0_norm, y0_norm = x0 / img_width, y0 / img_height
    x1_norm, y1_norm = x1 / img_width, y1 / img_height
    x2_norm, y2_norm = x2 / img_width, y2 / img_height
    x3_norm, y3_norm = x3 / img_width, y3 / img_height

    return [x0_norm, y0_norm, x1_norm, y1_norm, x2_norm, y2_norm, x3_norm, y3_norm]


def process_data(base_path: str) -> None:
    """
    Process data for a given base path.

    Args:
        base_path (str): The base path containing the data.

    Returns:
        None
    """
    gt_path = os.path.join(base_path, "gt/gt.txt")
    gameinfo_filepath = os.path.join(base_path, "gameinfo.ini")
    seqinfo_filepath = os.path.join(base_path, "seqinfo.ini")

    tracklets, id_map = parse_gameinfo(gameinfo_filepath)
    try:
        ball_id = [key for key, value in tracklets.items() if value == "ball;1"][0]
    except IndexError:
        ball_id = None

    seqinfo = parse_seqinfo(seqinfo_filepath)

    im_width = int(seqinfo["imwidth"])
    im_height = int(seqinfo["imheight"])
    ext = seqinfo["imext"]

    img_dir = os.path.join(base_path, f'{seqinfo["imdir"]}')

    label_dir = os.path.join(base_path, "labels")

    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(label_dir)

    img_paths = glob.glob(os.path.join(img_dir, f"*{ext}"))

    for img_path in tqdm(img_paths, desc="Processing images"):
        fname = os.path.basename(img_path).split(ext)[0]
        frame_id = int(fname)

        df = pd.read_csv(gt_path, header=None)
        df = df.rename(
            columns={
                0: "frame_id",
                1: "track_id",
                2: "x0",
                3: "y0",
                4: "width",
                5: "height",
                6: "gt",
            }
        )

        temp_df = df.groupby("frame_id").get_group(frame_id)

        label_spath = os.path.join(label_dir, f"{fname}.txt")
        with open(label_spath, "w", encoding='utf-8') as file:
            for _, row in temp_df.iterrows():
                if row["track_id"] != ball_id:
                    idt = id_map[row["track_id"]]
                    x0, y0, width, height = (
                        int(row["x0"]),
                        int(row["y0"]),
                        int(row["width"]),
                        int(row["height"]),
                    )
                    polygon_coords = rectangle_to_polygon_normalized(
                        x0, y0, width, height, im_width, im_height
                    )
                    file.write(
                        str(idt) + " " + " ".join(map(str, polygon_coords)) + "\n"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data in a directory.')
    parser.add_argument('--root_dir', '-i', required=True, type=str, help='Root directory path')
    args = parser.parse_args()

    root_dir = args.root_dir
    folders = os.listdir(root_dir)
    total_folders = len(folders)

    for index, folder in enumerate(folders, start=1):
        current_time = datetime.datetime.now(ist)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - Processing folder {index}/{total_folders}: {folder}")
        process_data(os.path.join(root_dir, folder))
