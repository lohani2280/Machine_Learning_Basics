import itertools
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb

# Mergeing train and test datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

target = train_df['final_status']
train_df.drop(['final_status', 'backers_count'],1,inplace=True)

df = pd.concat([train_df,test_df])

# working with merged datasets(train + test)
cols = ['project_id', 'name', 'desc', 'goal', 'keywords','disable_communication', 'country', 'currency', 'deadline',
       'state_changed_at', 'created_at', 'launched_at']
cols_drop = ['project_id']
cols_text = ['name','desc','keywords']
cols_time = ['deadline','state_changed_at','created_at','launched_at']
cols_categorical = ['country', 'currency', 'disable_communication']

# drop columns
# project_identity = df['project_id']
# df.drop(cols_drop,1,inplace = True)

# check missing values
print(df.isnull().sum())

# check datatypes of fetures
print(df.dtypes)

# fill missing values
df['desc'].fillna("None",inplace=True)
df['name'].fillna("None",inplace=True)

# set datetime
for col in cols_time:
	df[col] = pd.to_datetime(df[col],unit='s')

# Uniforming the goal(amount) feature to a unique currency(here, USD)
df_curr_factor = pd.read_csv('currency_factor.csv', index_col = 0)
extract_factor = lambda df_new,year,currency: df_new.loc[[currency],[year]].values[0][0]
new_goal = []
for i, row in df.iterrows():
	currency = row['currency']
	year = row['launched_at']
	new_goal.append(row['goal']*extract_factor(df_curr_factor,str(year.year),str(currency)))
df['new_goal'] = pd.Series(new_goal, index=df.index)
df.drop(['goal'],1,inplace = True)  


# LabelEncoding
for col in cols_categorical:
	le = LabelEncoder()
	le.fit(df[col])
	df[col] = le.transform(df[col])

# creating list with time difference 
diff = list(itertools.combinations(df[cols_time],2))
for i in diff:
	k = i[0] + ' - ' + i[1]
	df[k] = df[i[0]] - df[i[1]]


time_diff = ['deadline - state_changed_at', 'deadline - created_at',
             'deadline - launched_at', 'state_changed_at - created_at',
             'state_changed_at - launched_at', 'created_at - launched_at']

for i in time_diff:
	df[i] = df[i].dt.days

# Remove cols_time and creating new cols with day and month
for i,time_column in enumerate(cols_time):	
	df[str(time_column) + '_' + 'day'] = df[str(time_column)].dt.day	
	df[str(time_column) + '_' + 'month'] = df[str(time_column)].dt.month

df.drop(cols_time,1,inplace = True)

# No. of active projects while a project is launched
'''
no_active_projects = []
for i, _row in df.iterrows():
	launched_at = _row['launched_at']
	count = 0
	for _, row in df.iterrows():
		if (launched_at > row['launched_at']) and (launched_at < row['deadline']): 
			count += 1 
	no_active_projects.append(count)
			
df['no_active_projects'] = pd.Series(no_active_projects, index=df.index)
'''


#######################################################  TEXT DATA ANALYSIS  ######################################################


# working with merged datasets(train + test)
cols_text = ['name', 'keywords', 'desc']
cols_to_use = ['name','desc','keywords'] 
len_feats = ['name_len','desc_len','keywords_len']
count_feats = ['name_count','desc_count','keywords_count']

# Adding length feature
for i in np.arange(3):
    df[len_feats[i]] = df[cols_to_use[i]].apply(str).apply(len)

# Adding count feature
df['name_count'] = df['name'].str.split().str.len()
df['desc_count'] = df['desc'].str.split().str.len()
df['keywords_count'] = df['keywords'].str.split('-').str.len()

# This function cleans punctuations, digits and irregular tabs. Then converts the sentences to lower
def desc_clean(word):
    p1 = re.sub(pattern='(\W+)|(\d+)|(\s+)',repl=' ',string=word)
    p1 = p1.lower()
    return p1

# creating a full list of descriptions     
kickdesc = pd.Series(df['desc'].tolist()).astype(str)
kickdesc = kickdesc.map(desc_clean)

# Removing stopwords from desc(is,am,are,imself,therefore) with nltk. Converted kickdesc as a pandas series to list of list  
stop_words = set(stopwords.words('english'))
kickdesc = [[x for x in x.split() if x not in stop_words] for x in kickdesc]

# Stemming the words
'''
A stemming algorithm reduces the words "fishing", "fished", and "fisher" to the root word, "fish".
'''
stemmer = SnowballStemmer(language='english')
kickdesc = [[stemmer.stem(x) for x in x] for x in kickdesc]

# Further filtering kickdesc
kickdesc = [[x for x in x if len(x) > 2] for x in kickdesc]
kickdesc = [' '.join(x) for x in kickdesc]

print(kickdesc[0:10])

# CountVectorizer
'''
 let's say you have a CountVectorizer Object. Well, you call fit_transform for this object on some collection of strings.
 What it does is tokenize the strings and give you a vector for each string, 
 each dimension of which corresponds to the number of times a token is found in the corresponding string. 
 Most of the entries in all of the vectors will be zero, 
 since only the entries which correspond to tokens found in that specific string will have positive values, 
 but the vector is as long as the total number of tokens for the whole corpus.  
'''

'''
from sklearn.feature_extraction.text import CountVectorizer

texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)

print(cv.get_feature_names())
print(cv_fit.toarray())
#['bird', 'cat', 'dog', 'fish']
#[[0 1 1 1]
# [0 2 1 0]
# [1 0 0 1]
# [1 0 0 0]]

Each row in the array is one of your original documents (strings), 
each column is a feature (word), and the element is the count for that particular word and document. 
You can see that if you sum each column you'll get the correct number

print(cv_fit.toarray().sum(axis=0))
#[2 3 2 2]
'''
# Due to memory error, limited the number of features to 650
cv = CountVectorizer(max_features=650)
alldesc = cv.fit_transform(kickdesc).todense()
#create a data frame
combine = pd.DataFrame(alldesc)
combine.rename(columns= lambda x: 'variable_'+ str(x), inplace=True)

# Concatenate df and combine dataframe to get the complete preprocessed dataset
# df = df.loc[~df.index.duplicated(keep='first')]
# combine = combine.loc[~combine.index.duplicated(keep='first')]
# df = pd.concat([df,combine],axis=1)
# df = pd.concat([df,combine])
df = pd.concat([df, combine], axis=1, join_axes=[df.index])
print('DATAFRAME = After text analysis')
print('\n')
print(df.head(5))
print(df.dtypes)
print('STARTING XGBOOST')
df.drop(cols_text,1,inplace=True)


#####################################################    XGBOOST    #############################################################

train_data_df = df.iloc[0:108129,]
print('ayush',type(train_data_df))
test_data_df = df.iloc[108129:,]


train_data_df.drop(['project_id'], 1, inplace = True)

test_data_df_ids = pd.Series(test_data_df['project_id'])

test_data_df.drop(['project_id'], 1, inplace = True)


# for i in cols_categorical:
	# train_data_df[i] = train_data_df[i].astype(int)
	# test_data_df[i] = test_data_df[i].astype(int)

print(train_data_df.dtypes)
print(test_data_df.dtypes)	
print(train_data_df.head())	

xgb_tr = xgb.DMatrix(train_data_df, label=target, missing=np.NaN)
xgb_ts = xgb.DMatrix(test_data_df, missing=np.NaN)

############## Prediction model #################
param = {}
param['eta'] = 0.01
param['gamma'] = 1
param['max_depth'] = 6        #20
param['min_child_weight'] = 5
param['max_delta_step'] = 1000
param['subsample'] = 0.7
param['colsample_bytree '] = 0.7
param['lambda']=0.01
param['objective'] = 'binary:logistic'

num_round = 2000                    #2000

gbm = xgb.train(params=param,dtrain=xgb_tr, num_boost_round=num_round, verbose_eval=10)

gbm.save_model('500.model')

predictions = gbm.predict(xgb_ts)

# print type(test_pred[0])

idnames = df['project_id']
submission = pd.DataFrame({
            'project_id': idnames[108129:],
            'final_status': predictions })

# result = []
# for _,row in submission.iterrows():
# 	if row['final_status'] > 0.5:
# 		result.append(1)
# 	else:
# 		result.append(0)

# submission.drop('final_status',1,inplace = True)
# submission['final_status'] = pd.Series(result)			

submission.to_csv("submission_xgb_500.csv", index=False)
