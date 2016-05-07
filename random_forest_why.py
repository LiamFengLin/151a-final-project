import pandas as pd
from os.path import join
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

import pdb

pd.options.mode.chained_assignment = None 

USEFUL_COLUMNS = [
  "ResidentStatus",
  "EducationYears",
  "MonthOfDeath",
  "Sex",
  "AgeYears",
  "PlaceOfDeathAndDecedentsStatus",
  "MaritalStatus",
  "DayOfWeekOfDeath",
  "InjuryAtWork",
  "MannerOfDeath",
  "MethodOfDisposition",
  "Autopsy",
  "HispanicOriginRaceRecode",
  "CauseRecode13",
  "CauseRecode2",
  "ActivityCode",
  "PlaceOfInjury",
  "RaceRecode5"
]

LABEL_COLUMNS = ["CauseRecode13", "CauseRecode2"]

COLUMNS_TO_VECTORIZE = [
  "ResidentStatus",
  "Sex",
  "PlaceOfDeathAndDecedentsStatus",
  "MaritalStatus",
  "DayOfWeekOfDeath",
  "HispanicOriginRaceRecode",
  "MannerOfDeath",
  "MethodOfDisposition",
  "Autopsy",
  "ActivityCode",
  "PlaceOfInjury",
  "InjuryAtWork",
  "RaceRecode5"
]

IMPUTED_COLUMN_TO_VALUE = {
  "Education2003Revision" : 9,
  "AgeRecode27": 27,
  "PlaceOfDeathAndDecedentsStatus": 9,
  "MaritalStatus": "U",
  "DayOfWeekOfDeath": 9,
  "InjuryAtWork": "U",
  "MannerOfDeath": 0,
  "Autopsy": "U",
  "PlaceOfInjury": 9
}


CAUSE_CODE_MAP_BINARY = {
  1: "A",
  2: "A",
  3: "A",
  4: "A",
  5: "A",
  6: "A",
  7: "A",
  8: "A",
  9: "A",
  10: "A",
  11: "A",
  12: "A",
  13: "A",
  14: "A",
  15: "A",
  16: "A",
  17: "A",
  18: "A",
  19: "A",
  20: "A",
  21: "A",
  22: "A",
  23: "A",
  24: "A",
  25: "A",
  26: "A",
  27: "A",
  28: "A",
  29: "A",
  30: "A",
  31: "A",
  32: "A",
  33: "A",
  34: "A",
  35: "A",
  36: "A",
  37: "A",
  38: "X",
  39: "X",
  40: "X",
  41: "X",
  42: "X"
}


CAUSE_CODE_MAP = {
  1: "A",
  2: "A",
  3: "B", # HIV
  4: "C", # Cancer
  5: "C",
  6: "C",
  7: "C",
  8: "C",
  9: "C",
  10: "C",
  11: "C",
  12: "C",
  13: "C",
  14: "C",
  15: "C",
  16: "E", # Diabetes
  17: "G", # Alzheimer
  18: "I", # Heart Disease
  19: "I",
  20: "I",
  21: "I",
  22: "I",
  23: "I",
  24: "I",
  25: "I",
  26: "I",
  27: "J",  # Respiratory-related
  28: "J",
  29: "K",
  30: "K",
  31: "N",
  32: "O",
  33: "R",
  34: "R", # Infant-related
  35: "R",
  36: "R",
  37: "OA",
  38: "X", # External
  39: "X",
  40: "X",
  41: "X",
  42: "X"
}

EDUCATION_TO_YEARS_MAP = {
  1: 4,
  2: 10,
  3: 12,
  4: 13,
  5: 14,
  6: 16,
  7: 18,
  8: 23
}

AGE_TO_YEARS_MAP = {
  1: 0.08,
  2: 0.5,
  3: 1,
  4: 2,
  5: 3,
  6: 4,
  7: 7,
  8: 12,
  9: 17,
  10: 22,
  11: 27,
  12: 32,
  13: 37,
  14: 42,
  15: 47,
  16: 52,
  17: 57,
  18: 62,
  19: 67,
  20: 72,
  21: 77,
  22: 82,
  23: 87,
  24: 92,
  25: 97,
  26: 100
}

def benchmark(predicted_labels, true_labels):
  err_indices = [i for i, labels in enumerate(zip(predicted_labels, true_labels)) if labels[0] != labels[1] ]
  err_rate = len(err_indices) * 1.0 / true_labels.shape[0]
  return err_rate, err_indices

def plot_feature_importances(feature_names, scores, which_labels):
  indices = np.argsort(scores)[::-1]
  ranked_scores = scores[indices][:10]
  ranked_features = feature_names[indices][:10]

  plt.figure()
  plt.title("Feature Importances, Random Forest")
  plt.bar(range(len(ranked_features)), ranked_scores, color="r", align="center")
  plt.xticks(range(len(ranked_features)), ranked_features, rotation="vertical")
  plt.tight_layout()
  plt.savefig("feature_importances_rf_%s.png" % which_labels, format='png', dpi=500)  

def plot_oob_error_depth(depths, oob_scores, title):

  plt.figure()
  plt.plot(depths, oob_scores)
  plt.title("Out-of-Bag Error against Depth of Trees in Random Forest")
  plt.xlabel("Depth")
  plt.ylabel("Out-of-Bag Error")
  plt.tight_layout()
  plt.savefig("oob_error_against_depth_why_%s.png" % title, format='png', dpi=500)  

def plot_oob_error_n_tress(n_trees, oob_scores, title):

  plt.figure()
  plt.plot(n_trees, oob_scores)
  plt.title("Out-of-Bag Error against Number of Trees in Random Forest")
  plt.xlabel("NUmber of Trees")
  plt.ylabel("Out-of-Bag Error")
  plt.tight_layout()
  plt.savefig("oob_error_against_n_trees_why_%s.png" % title, format='png', dpi=500)  


class Preprocess:
  def __init__(self, records, which_labels = "binary"):

    self.which_labels = which_labels

    self.records = records
    self.filter_columns()
    self.vectorize()

    features, labels = self.separate_data()
    self.features = features
    self.labels = labels

  def filter_columns(self):
    self.records = self.records[USEFUL_COLUMNS]

  def vectorize(self):
    self.records = pd.get_dummies(self.records, columns = COLUMNS_TO_VECTORIZE)

  def separate_data(self):
    feature_columns = list(self.records.columns.values.tolist())
    feature_columns.remove(LABEL_COLUMNS[0])
    feature_columns.remove(LABEL_COLUMNS[1])
    if self.which_labels == "binary":
      return self.records[feature_columns], self.records[LABEL_COLUMNS[1]]
    elif self.which_labels == "13":
      return self.records[feature_columns], self.records[LABEL_COLUMNS[0]]
    else:
      raise ValueError("Invalid code")

def cross_validate_depth(label_group):

  sample = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_train.csv"))
  test = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_test.csv"))

  preprocessed = Preprocess(sample, which_labels = label_group)

  depths = (2, 40, 60, 80)
  oob_scores = []
  for depth in depths:
    rf = RandomForestClassifier(n_estimators = 80, criterion = "entropy", oob_score = True, bootstrap = True, max_features = 'sqrt', max_depth = depth)
    rf.fit_transform(X = preprocessed.features, y = preprocessed.labels.values.ravel())

    score = 1.0 - rf.oob_score_
    
    oob_scores.append(score)
    print "Out-of-Bag Error for Depth %s: %s" % (depth, score)

  plot_oob_error_depth(depths, oob_scores, label_group)

def cross_validate_number_of_trees(label_group):

  sample = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_train.csv"))
  test = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_test.csv"))

  preprocessed = Preprocess(sample, which_labels = label_group)

  n_trees = (5, 10, 30, 60, 80)
  oob_scores = []
  for n_tree in n_trees:
    rf = RandomForestClassifier(n_estimators = n_tree, criterion = "entropy", oob_score = True, bootstrap = True, max_features = 'sqrt', max_depth = 40)
    rf.fit_transform(X = preprocessed.features, y = preprocessed.labels.values.ravel())

    score = 1.0 - rf.oob_score_
    
    oob_scores.append(score)
    print "Out-of-Bag Error for Number of Trees %s: %s" % (n_tree, score)

  plot_oob_error_n_tress(n_trees, oob_scores, label_group)

def predict_on_test_set(label_group):
  sample = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_train.csv"))
  test = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_test.csv"))

  preprocessed = Preprocess(sample, which_labels = label_group)

  rf = RandomForestClassifier(n_estimators = 80, criterion = "entropy", bootstrap = True, max_features = 'sqrt', max_depth = 40)
  rf.fit_transform(X = preprocessed.features, y = preprocessed.labels.values.ravel())
  test_preprocessed = Preprocess(test, which_labels = label_group)
  predicted_labels = rf.predict(test_preprocessed.features)
  error_rate, _ = benchmark(predicted_labels.ravel(), test_preprocessed.labels.values)

  plot_feature_importances(preprocessed.features.columns.values, rf.feature_importances_, label_group)

if __name__ == "__main__":

  ROOT = os.path.dirname(__file__)
  RECORDS_FILE_PATH = join(ROOT, "data", "DeathRecords")
  SAMPLES_FILE_PATH = join(ROOT, "samples")

  cross_validate_depth("13")
  cross_validate_depth("binary")
  cross_validate_number_of_trees("13")
  cross_validate_number_of_trees("binary")

  predict_on_test_set("13")
  predict_on_test_set("binary")