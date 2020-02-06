import pandas as pd
import torch
import numpy as np
from typing import List
import ast
import os
import pathlib
from tqdm import tqdm
import pdb
import subprocess


def decode_dt(code: str) -> List:
    # capital letters are 65 - 90 inclusive
    # lower case are 97 - 122 inclusive
    capital_letters = [chr(i) for i in range(65, 91)]
    lower_case = [chr(i) for i in range(97, 123)]

    atlas = {}
    number = -2

    for letter in capital_letters:
        atlas[letter] = number
        number -= 2

    number = 2
    for letter in lower_case:
        atlas[letter] = number
        number += 2

    dt_code = []
    for letter in code:
        dt_code.append(atlas[letter])

    return dt_code

def polystr_to_polylist(polystr: str) -> List:
    polylist = ast.literal_eval(polystr)
    return polylist

def poly_to_str(poly: List) -> str:
    out_str = ""
    for i, (exp, coeff) in enumerate(poly):
        out_str += f" {coeff} x^ {exp} "
        if i != (len(poly) - 1):
            out_str += "+"
    return out_str

def poly_representation_padding(df: pd.DataFrame) -> [List, List]:
    """This function is mostly for reproducability purposes. There isn't much chance it will get used again"""

    out = []
    for poly_name in ['Jones_polynomial', 'Alexander_polynomial']:
        poly = df[poly_name].apply(lambda x: eval(x)).values
        poly_list = []
        for i in poly:
            poly_list.append(torch.tensor(list(i)))
        out.append(torch.nn.utils.rnn.pad_sequence(poly_list))

    return out



def put_str_repr_on_csv():
    """
    For all the .csv files in this directory, Take the Jones and Alexander
    polynomial list representations and change them to string representations.
    The file that is produced is a dataframe with these string representations
    appended to the original file.
    """
    for file_name in tqdm(os.listdir("./raw_data")):
        file = pathlib.Path(f"./raw_data/{file_name}")
        if file.suffix == ".csv" and "str" not in file.stem:
            df = pd.read_csv("./raw_data/"+file_name)
            df['jones_str'] = df['Jones_polynomial'].apply(lambda x: eval(x)).transform(lambda x: poly_to_str(x))
            df['alexander_str'] = df['Alexander_polynomial'].apply(lambda x: eval(x)).transform(lambda x: poly_to_str(x))
            df.to_csv(f"{file.stem}_str.csv", index=False)

def format_csv_to_txt_for_openmt(file_name):
    df = pd.read_csv(file_name)
    file = pathlib.Path(f"{file_name}")
    with open(f"./data/{file.stem}_jones.txt", "w") as f:
        for poly in tqdm(df['jones_str']):
            f.write(poly)
            f.write("\n")
    with open(f"./data/{file.stem}_dt.txt", "w") as f:
        for code in tqdm(df['dt_code']):
            code = " ".join([char for char in code])
            f.write(code)
            f.write("\n")

def get_vocab():
    dt_vocab = set()
    jones_vocab = set()
    for file_name in tqdm(os.listdir("./data")):
        file = pathlib.Path(f"./data/{file_name}")

        s = ""
        if ".txt" == file.suffix:
            with open(file, "r") as f:
                s = f.read()
            s = s.replace("\n", " ")
            s = s.split(" ")

        if "_dt" in file.stem:
            dt_vocab = dt_vocab.union(set(s))

        elif "_jones" in file.stem:
            jones_vocab = jones_vocab.union(set(s))

    with open("./data/dt_vocab.txt", "w") as f:
        for item in dt_vocab:
            f.write(item)
            f.write("\n")

    with open("./data/jones_vocab.txt", "w") as f:
        for item in jones_vocab:
            f.write(item)
            f.write("\n")

def list_builder(li):
    li_str = "{"
    for i in li:
        li_str += str(i) + ","
    li_str = li_str[:-1] + "}"
    return li_str

def calculate_accuracy(ground_truth_dt: List[str], predicted_dt: List[str], sage_path="/home/andrew/Documents/SageMath/sage") -> int:
    prediction_bools = []

    for i, code in tqdm(enumerate(predicted_dt)):
        gt_code = ground_truth_dt[i]
        code1 = decode_dt(code.replace(" ", "").replace("\n", ""))
        code2 = decode_dt(gt_code.replace(" ", "").replace("\n", ""))
        if code1 != code2:
            # result = subprocess.run(["bash", sage_path, "sage_accuracy.py", f"\"{code1}\"", f"\"{code2}\""], stdout=subprocess.PIPE).stdout
            result = subprocess.run(["./wolf_accuracy.wls", f"\"{list_builder(code1)}\"", f"\"{list_builder(code2)}\""], stdout=subprocess.PIPE).stdout
            result = result.decode("utf-8")
            result = result.split("\n")[-2]
            is_same_knot = eval(result)
        else:
            print("skip")
            is_same_knot = True
        prediction_bools.append(is_same_knot)
        if i % 25 == 0:
            print(f"\nNum True: {sum(prediction_bools)}\nNum False:{len(prediction_bools) - sum(prediction_bools)}\nTotal:{len(prediction_bools)}")

    return sum(prediction_bools)/len(prediction_bools)

def parallel_accuracy():
    gt_list, pred_list = process_files("./data/KnotTableLarge_19a_str_dt.txt", "pred.txt")
    n = len(pred_list)
    gt_list = gt_list[:n]
    data = [(gt_list[:n//8], pred_list[:n//8]),
            (gt_list[1 * n//8:2 * n//8], pred_list[1 * n//8:2 * n//8]),
            (gt_list[2 * n//8:3 * n//8], pred_list[2 * n//8:3 * n//8]),
            (gt_list[3 * n//8:4 * n//8], pred_list[3 * n//8:4 * n//8]),
            (gt_list[4 * n//8:5 * n//8], pred_list[4 * n//8:5 * n//8]),
            (gt_list[5 * n//8:6 * n//8], pred_list[5 * n//8:6 * n//8]),
            (gt_list[6 * n//8:7 * n//8], pred_list[6 * n//8:7 * n//8]),
            (gt_list[7 * n//8:8 * n//8], pred_list[7 * n//8:8 * n//8])]


def process_files(gt_dt_filename: str, pred_dt_filename: str):
    with open(gt_dt_filename, "r") as f:
        gt_list = f.readlines()

    with open(pred_dt_filename, "r") as f:
        pred_list = f.readlines()

    return gt_list, pred_list



if __name__ == "__main__":
    # print("adding column to csv")
    acc = calculate_accuracy(*process_files("./data/KnotTableLarge_19a_str_dt.txt", "pred.txt"))
    print(acc)
    # put_str_repr_on_csv()

    # print("write text file")
    # for file in os.listdir("./"):
        # file = pathlib.Path(f"./{file}")
        # if file.suffix == ".csv" and "str" in file.stem:
            # format_csv_to_txt_for_openmt(file)

