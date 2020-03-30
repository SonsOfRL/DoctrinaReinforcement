import numpy as np
import csv
import os
import re


class DiskWriter():

    def __init__(self, write_path, fields):
        while os.path.exists(write_path):
            match = re.findall("\([0-9]+\).csv$", write_path)
            if match:
                number = str(int(match[-1][1:-5]) + 1)
                write_path = re.sub(
                    "\([0-9]+\).csv$", "({}).csv".format(number), write_path)
            else:
                write_path = re.sub(".csv$", "(1).csv", write_path)
        self.write_path = write_path
        self.fields = fields

        self.file = open(self.write_path, "a")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()

    def __call__(self, field_dict):
        self.writer.writerow(field_dict)

    def __exit__(self):
        self.file.close()

    @staticmethod
    def readall(read_path, field, map_fn=lambda x: x):
        with open(read_path, "r") as f:
            reader = csv.DictReader(f)

            return [map_fn(row[field]) for row in reader]


class PrintWriter():

    def __init__(self, end_line="\n", flush=False):
        self.end_line = end_line
        self.flush = flush

    def __call__(self, field_dict):
        print(
            ", ".join(key.format(value) for key, value in field_dict.items()),
            end=self.end_line,
            flush=self.flush,
        )


class MultiWriter():

    def __init__(self, writers):
        self.writers = writers

    def __call__(self, field_dict):
        for w in self.writers:
            w(field_dict)
