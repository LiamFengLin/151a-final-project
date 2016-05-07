import pandas as pd
from os.path import join
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import random_forest_why


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
  "CauseRecode39",
  "CauseRecode2",
  "ActivityCode",
  "PlaceOfInjury",
  "RaceRecode5"
]

LABEL_COLUMN = "AgeYears"

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
  "RaceRecode5",
  "CauseRecode39",
  "CauseRecode2"
]

IMPUTED_COLUMN_TO_VALUE = random_forest.IMPUTED_COLUMN_TO_VALUE

def benchmark(predicted_labels, true_labels):
  err_indices = [i for i, labels in enumerate(zip(predicted_labels, true_labels)) if labels[0] != labels[1] ]
  err_rate = len(err_indices) * 1.0 / true_labels.shape[0]
  return err_rate, err_indices

def plot_feature_importances(feature_names, scores):
  indices = np.argsort(scores)[::-1]
  ranked_scores = scores[indices][:10]
  ranked_features = feature_names[indices][:10]

  plt.figure()
  plt.title("Feature Importances, Random Forest")
  plt.bar(range(len(ranked_features)), ranked_scores, color="r", align="center")
  plt.xticks(range(len(ranked_features)), ranked_features, rotation="vertical")
  plt.tight_layout()
  plt.savefig("feature_importances_rf_when.png", format='png', dpi=500)  

class Preprocess:
  def __init__(self, records):

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
    feature_columns.remove(LABEL_COLUMN)
    return self.records[feature_columns], self.records[LABEL_COLUMN]

def MSE_prediction(predicted_values, true_values):
  return np.sqrt(np.sum((predicted_values - true_values) ** 2) * 1.0 / len(predicted_values))

def cross_validate_depth():

  sample = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_train.csv"))

  preprocessed = Preprocess(sample)

  depths = (2, 40, 60, 80)
  oob_scores = []
  for depth in depths:
    rf = RandomForestRegressor(n_estimators = 60, criterion = "mse", bootstrap = True, oob_score = True, max_features = 'sqrt', max_depth = depth)
    rf.fit_transform(X = preprocessed.features, y = preprocessed.labels.values.ravel())

    score = 1.0 - rf.oob_score_
    
    oob_scores.append(score)
    print "Out-of-Bag Error for Depth %s: %s" % (depth, score)

  pdb.set_trace()

  plot_oob_error_depth(depths, oob_scores)

def plot_oob_error_depth(depths, oob_scores):

  plt.figure()
  plt.plot(depths, oob_scores)
  plt.title("Out-of-Bag Error against Depth of Trees in Random Forest")
  plt.xlabel("Depth")
  plt.ylabel("Out-of-Bag Error")
  plt.tight_layout()
  plt.savefig("oob_error_against_depth_when.png", format='png', dpi=500)  

def predict_on_test():
  sample = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_train.csv"))
  test = pd.read_csv(join(SAMPLES_FILE_PATH, "sample_test.csv"))

  preprocessed = Preprocess(sample)

  rf = RandomForestRegressor(n_estimators = 100, criterion = "mse", bootstrap = True, max_features = 'sqrt', depth = 40)
  rf.fit_transform(X = preprocessed.features, y = preprocessed.labels.values.ravel())
  test_preprocessed = Preprocess(test)
  predicted_values = rf.predict(test_preprocessed.features)
  error_rate, _ = benchmark(predicted_values.ravel(), test_preprocessed.labels.values)

  print "Mean Square Prediction Erorr = %s" % MSE_prediction(predicted_values, test_preprocessed.labels)

  plot_feature_importances(preprocessed.features.columns.values, rf.feature_importances_)

ROOT = os.path.dirname(__file__)
RECORDS_FILE_PATH = join(ROOT, "data", "DeathRecords")
SAMPLES_FILE_PATH = join(ROOT, "samples")

if __name__ == "__main__":
  cross_validate_depth()
  predict_on_test()