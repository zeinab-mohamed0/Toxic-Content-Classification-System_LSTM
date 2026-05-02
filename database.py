import csv
import os

FILE_NAME = "database.csv"

def save_result(Caption, Label):
    file_exists = os.path.isfile(FILE_NAME)

    with open(FILE_NAME, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)


        writer.writerow([Caption, Label])