import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
df = pd.read_csv('car_insurance.csv')
df.head()
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].mean())
df['annual_mileage'] = df['annual_mileage'].fillna(df['annual_mileage'].mean())
models = []
features = df.drop(['outcome', 'id'], axis = 1).columns.tolist()
for col in features:
    model = logit(f"outcome ~ {col}", data=df).fit()
    print(f'{col}')
    models.append(model)
accuracies = []
for i in range(len(models)):
    conf_matrix = models[i].pred_table()
    tn = conf_matrix[0,0]
    tp = conf_matrix[1,1]
    fn = conf_matrix[1,0]
    fp = conf_matrix[0,1]
    accuracy = (tn + tp)/(tn+tp+fn+fp)
    accuracies.append(accuracy)
best_acc_index = accuracies.index(max(accuracies))
best_feature = features[best_acc_index]
best_accuracy = accuracies[best_acc_index]
best_feature_df = pd.DataFrame({'best_feature': best_feature, 
                               'best_accuracy': best_accuracy}, index = [0])
model = logit(f"outcome ~ {col}", data=df).fit(disp=0)
print(best_feature_df.head())
