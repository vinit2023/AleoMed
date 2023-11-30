from pathlib import Path

import numpy
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import logging


def map_dna_sym_to_digit(symbol):
  if symbol == "T":
    return 0
  if symbol == "C":
    return 1
  if symbol == "G":
    return 2
  if symbol == "A":
    return 3

  return -1

def read_tcga(filename):
  with open(filename, "r") as f:
    lines = f.readlines()
    array_2d = np.zeros(shape=(len(lines), 15), dtype=np.uint8)
    for i in range(len(lines)):
      line_str = str(lines[i]).strip()
      for j in range(len(line_str)):
        symbol = line_str[j]
        digit = map_dna_sym_to_digit(symbol)
        if digit > -1:
          array_2d[i][j] = digit
  return array_2d


def read_output(filename):
  with open(filename, "r") as f:
    lines = f.readlines()
    array_of_ints = np.zeros(len(lines), dtype=np.uint8)
    for i in range(len(lines)):
      line_txt = str(lines[i]).strip()
      percentage = int(line_txt)
      array_of_ints[i] = percentage
  return array_of_ints


logger = logging.getLogger()
logger.setLevel(logging.INFO)

training_input = read_tcga("tmp/dna/inputs.txt")
training_output = read_output("tmp/dna/outputs.txt")


clf = DecisionTreeClassifier(max_depth=15)
clf = clf.fit(training_input, training_output)

test_line = "ACTTCGGCTCACTCC"

test_digits = np.zeros((1, 15), dtype=np.uint8)
for i in range(len(test_line)):
  digit = map_dna_sym_to_digit(test_line[i])
  if digit > -1:
    test_digits[0][i] = digit

print(f"test result: {clf.predict(test_digits)}")


from zkml import LeoTranspiler


lt = LeoTranspiler(model=clf, validation_data=test_digits)

leo_project_name = "tree_dna_1"
lt.to_leo(
    path=Path("/Users/ilyabelkin/PycharmProjects/pythonProject/tmp/dna"), project_name=leo_project_name, fixed_point_scaling_factor=16
)
