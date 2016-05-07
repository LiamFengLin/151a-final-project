import pandas as pd
from os.path import join
import os
import numpy as np
import matplotlib.pyplot as plt

import pdb

pd.options.mode.chained_assignment = None 


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

INFANT_AGE_TO_YEARS_MAP = {
  0:1.0,
  1:0.000114155,
  2:0.001712328,
  3:1.0 / 365,
  4:2.0 / 365,
  5:3.0 / 365,
  6:4.0 / 365,
  7:5.0 / 365,
  8:6.0 / 365,
  9:10.0 / 365,
  10:17.0 / 365,
  11:24.0 / 365,
  12:1.0 / 12,
  13:2.0 / 12,
  14:3.0 / 12,
  15:4.0 / 12,
  16:5.0 / 12,
  17:6.0 / 12,
  18:7.0 / 12,
  19:8.0 / 12,
  20:9.0 / 12,
  21:10.0 / 12,
  22:11.0 / 12
}


class Preprocess:
  def __init__(self, records):
    self.records = records
    self.clean()
    self.filter_records()
    self.map_age_to_continuous()
    self.impute_categorical()
    self.map_education()
    self.map_cause()
    
  def clean(self):
    if np.sum(self.records["Autopsy"] == "n") != 0:
      self.records["Autopsy"][self.records["Autopsy"] == "n"] = "N"
    if np.sum(self.records["Autopsy"] == "y") != 0:
      self.records["Autopsy"][self.records["Autopsy"] == "y"] = "Y"

  def filter_records(self):
    self.records = self.records.loc[self.records['EducationReportingFlag'] == 1]
    self.records = self.records.loc[~((self.records['AgeType'] == 1) & (self.records['Age'] > 200))]

  def impute_categorical(self):
    for col in self.records.columns.values:

      if col in IMPUTED_COLUMN_TO_VALUE.keys(): 
        substitute_value = self.records[col].value_counts().idxmax()
        self.records[col][self.records[col] == IMPUTED_COLUMN_TO_VALUE[col]] = substitute_value

  def map_education(self):
    col = self.records['Education2003Revision'].tolist()
    replacement = [EDUCATION_TO_YEARS_MAP[elem] for elem in col]
    self.records['EducationYears'] = replacement

  def map_cause(self):
    col = self.records['CauseRecode39'].tolist()
    replacement = [CAUSE_CODE_MAP_BINARY[elem] for elem in col]
    self.records['CauseRecode2'] = replacement

    replacement = [CAUSE_CODE_MAP[elem] for elem in col]
    self.records['CauseRecode13'] = replacement

  def map_age_to_continuous(self):
    age_types = np.array(self.records['AgeType'].tolist())
    ages = np.array(self.records['Age'].tolist()).astype(np.float64)

    for i in range(self.records.shape[0]):
      age_type, age = age_types[i], ages[i] * 1.0
      if age_types[i] == 1:
        continue
      elif age_types[i] == 2:
        ages[i] = age / 12
      elif age_types[i] == 4:
        ages[i] = age / 365
      elif age_types[i] == 5:
        ages[i] = age / (365 * 24)
      elif age_types[i] == 6:
        ages[i] = age / (365 * 24 * 60)
      elif age_types[i] == 9:
        ages[i] = 0.0
      else:
        raise ValueError("Invalid code")
    mean_age = np.sum(ages) * 1.0 / np.sum(age_types != 9)

    print "mean age: %s" % mean_age
    
    for j in range(self.records.shape[0]):
      if age_types[j] == 9:
        ages[j] = mean_age

    self.records['AgeYears'] = ages

def visualize_missing_values(records):
  key_names = IMPUTED_COLUMN_TO_VALUE.keys()
  percentages = [np.sum(records[col] == IMPUTED_COLUMN_TO_VALUE[col]) * 1.0 / records[col].size for col in key_names]

  plt.figure()
  plt.title("Percentage of Missing Values by Column")
  plt.ylabel("Percentage")
  plt.xticks(range(len(key_names)), key_names, rotation="vertical")
  plt.scatter(x = range(len(key_names)), y = percentages)
  plt.tight_layout()
  plt.savefig("missing_values.png", format='png', dpi=500)  

ROOT = os.path.dirname(__file__)
RECORDS_FILE_PATH = join(ROOT, "data", "DeathRecords")
SAMPLES_FILE_PATH = join(ROOT, "samples")

if __name__ == "__main__":

  main_records = pd.read_csv(join(RECORDS_FILE_PATH, "DeathRecords.csv"))

  visualize_missing_values(main_records)

  main_records = main_records.sample(frac=1).reset_index(drop=True)

  preprocessed = Preprocess(main_records)

  nrows = preprocessed.records.shape[0]

  train_large = preprocessed.records.loc[0:nrows / 2]
  test_large = preprocessed.records.loc[nrows / 2:]
  train_large.to_csv(join(SAMPLES_FILE_PATH, "sample_train.csv"), index = False)
  test_large.to_csv(join(SAMPLES_FILE_PATH, "sample_test.csv"), index = False)

  train = preprocessed.records.loc[0:nrows / 8]
  test = preprocessed.records.loc[nrows / 8:nrows * 2 / 8]
  train.to_csv(join(SAMPLES_FILE_PATH, "sample_train_medium.csv"), index = False)
  test.to_csv(join(SAMPLES_FILE_PATH, "sample_test_medium.csv"), index = False)