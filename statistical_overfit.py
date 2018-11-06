import csv
import gzip
import numpy as np
import pylab as pl
from sklearn
import linear_model, cross_validation, svm, datasets
from sklearn.metrics
import mean_squared_error

train_filename = 'train.csv.gz'
test_filename = 'test.csv.gz'
pred_filename = 'prediction.csv'
train_data = []

with gzip.open(train_filename, 'r') as train_fh:
  train_csv = csv.reader(train_fh, delimiter = ',', quotechar = '"')

next(train_csv, None)
for row in train_csv:
  smiles = row[0]
features = np.array([float(x) for x in row[1: 257]])
gap = float(row[257])

train_data.append({
  'smiles': smiles,
  'features': features,
  'gap': gap
})

test_data = []

with gzip.open(test_filename, 'r') as test_fh:
  test_csv = csv.reader(test_fh, delimiter = ',', quotechar = '"')

next(test_csv, None)
for row in test_csv:
  id = row[0]
smiles = row[1]
features = np.array([float(x) for x in row[2: 258]])

test_data.append({
  'id': id,
  'smiles': smiles,
  'features': features
})

test_data_100000 = test_data[0: 100000]
train_data_100000 = train_data[0: 100000]

def target_select(data, key):
  feat = []
  for i in range(0, len(data)):
	feat.append(data[i: i + 1][0][key])
  return feat

feat_train = []

for i in range(0, len(train_data_100000)):
  feat_train.append(train_data_100000[i]['features'])

target = np.array(target_select(train_data_100000, 'gap'))
model_linear = linear_model.LinearRegression()
model_ridge = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0])
model_lasso = linear_model.LassoCV()
model_elastic = linear_model.ElasticNetCV()
model_bayesRidge = sklearn.linear_model.BayesianRidge()
model_svm = sklearn.svm.SVC()
cross_validation.cross_val_score(model_linear, feat_train, target, cv = 10, scoring = 'mean_squared_error')
model_linear.fit(feat_train, target)

feat_test = []

for i in range(0, len(test_data_100000)):
  feat_test.append(test_data_100000[i]['features'])

for j in range(0, len(test_data_100000)):
  test_data_100000[j: j + 1][0]['prediction'] = model_linear.predict(test_data_100000[j]['features'])

with open(pred_filename, 'w') as pred_fh:
  pred_csv = csv.writer(pred_fh, delimiter = ',', quotechar = '"')
pred_csv.writerow(['Id', 'Prediction'])

for datum in test_data_100000:
  pred_csv.writerow([datum['id'], datum['prediction']])