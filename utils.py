import pandas as pd
from typing import List
import sys

file_name = sys.argv[1]

df = pd.read_csv(file_name)

print(df)


def string_to_dt(code: str) -> List:
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

def jones_to_str(poly: List) -> str:
    jones_str = ""
    for i, (coeff, exp) in enumerate(poly):
        jones_str += f" {coeff}x^{exp} "
        if i != (len(poly) - 1):
            jones_str += "+"
    return jones_str

print(string_to_dt(df.iloc[0]["dt_code"]))
print(jones_to_str(eval(df.iloc[0]["Jones_polynomial"])))
