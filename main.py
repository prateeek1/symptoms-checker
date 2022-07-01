#!/usr/bin/env python
# coding: utf-8

# In[2]:


import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from decimal import Decimal
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[20]:


df = pd.read_csv('./dataset.csv')


# In[21]:


df.head()


# In[22]:


len(df)


# In[42]:


unique = []
for i in range(0, 4920):
    row = np.asarray(df[i:i+1])
    for i in range(1, len(row[0])):
        if(pd.isna(row[0][i])):
            break
        row[0][i] = row[0][i].replace('_', ' ')
        if row[0][i] not in unique:
            unique.append(row[0][i])


# In[43]:


print(unique)


# In[25]:


# In[45]:


df_upd = pd.DataFrame(columns=unique)
df_upd.head()


# In[55]:


df_upd = df_upd.astype('int')


# In[29]:


# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder

# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(unique)
# print(integer_encoded)


# In[30]:


# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded[1])


# In[56]:


disease = []
for i in range(0, 4920):
    row = np.asarray(df[i:i+1])
    col = [0]*131
    symptom = []
    disease.append(row[0][0])
    for j in range(1, len(row[0])):
        if(pd.isna(row[0][j])):
            break
        symptom.append(row[0][j])
    for j in range(0, len(unique)):
        if unique[j] in symptom:
            col[j] = 1

    df_upd.loc[i] = col


# In[57]:


df_disease = pd.DataFrame(disease)
print(df_disease)


# In[60]:


df_concat = pd.concat([df_disease, df_upd], axis=1)
df_concat.drop_duplicates(keep='first', inplace=True)
df_concat.head()


# In[59]:


print(df_concat.dtypes)


# In[ ]:


# In[61]:


Y = df_concat[0]
X = df_concat[unique]


# In[62]:


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=101)


# In[64]:


len(x_train), len(y_train)


# In[65]:


len(x_test), len(y_test)


# In[ ]:


# In[69]:


dt = DecisionTreeClassifier()
clf_dt = dt.fit(x_train, y_train)
clf = clf_dt.predict(x_test)
acc_mnb = round(Decimal(accuracy_score(y_test, clf) * 100), 2)
print(acc_mnb)


# In[76]:


mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)
mnb_pred = mnb.predict(x_test)
acc_mnb = round(Decimal(accuracy_score(y_test, mnb_pred) * 100), 2)
print(acc_mnb)


# In[80]:


rfc = RandomForestClassifier()
rfc = rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
acc_rfc = round(Decimal(accuracy_score(y_test, rfc_pred) * 100), 2)
print(acc_rfc)


# In[100]:


input = [0]*131
input[121] = 1
res = rfc.predict([input])
res[0]
print(desc[res[0]])


# In[93]:


dff = pd.read_csv('./symptom_description.csv')


# In[101]:


dff.head()
desc = {}
for i in range(0, len(dff)):
    desc[dff['Disease'][i]] = dff['Description'][i]


# In[6]:


dff = pd.read_csv('./symptom_precaution.csv')
for i in range(len(dff['Precaution_4'])):
    if pd.isna(dff['Precaution_4'][i]):
        dff['Precaution_4'][i] = "Exercise and be physically fit"
        dff['Precaution_3'][i] = "Keep check and control the cholesterol levels"
        dff['Precaution_1'][i] = "Manage Stress"
        dff['Precaution_2'][i] = "Get proper sleep"
    else:
        s = dff['Precaution_4'][i]
        s = s[0].upper()+s[1:]
        dff['Precaution_4'][i] = s
        s = dff['Precaution_3'][i]
        s = s[0].upper()+s[1:]
        dff['Precaution_3'][i] = s
        s = dff['Precaution_2'][i]
        s = s[0].upper()+s[1:]
        dff['Precaution_2'][i] = s
        s = dff['Precaution_1'][i]
        s = s[0].upper()+s[1:]
        dff['Precaution_1'][i] = s


# In[9]:


pre1 = {}
pre2 = {}
pre3 = {}
pre4 = {}
for i in range(0, len(dff)):
    pre1[dff['Disease'][i]] = dff['Precaution_1'][i]
    pre2[dff['Disease'][i]] = dff['Precaution_2'][i]
    pre3[dff['Disease'][i]] = dff['Precaution_3'][i]
    pre4[dff['Disease'][i]] = dff['Precaution_4'][i]


# In[23]:


joblib.dump(rfc, 'rfc.pkl')


# In[24]:


def get_symptom():
    return unique


# In[25]:


def get_description(disease):
    return desc[disease]


# In[26]:


def get_precaution1(disease):
    return pre1[disease]


def get_precaution2(disease):
    return pre2[disease]


def get_precaution3(disease):
    return pre3[disease]


def get_precaution4(disease):
    return pre4[disease]


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
