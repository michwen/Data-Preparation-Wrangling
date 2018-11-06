import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

df = pd.read_csv("test.csv")
type(df)

print df.head()
print df.describe()

print data.head()

train_cols = data.columns[1: ]
logit = sm.Logit(data['score'], data[train_cols])

result = logit.fit()

print result.summary()
print result.conf_int()
print np.exp(result.params)

params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']

print np.exp(conf)