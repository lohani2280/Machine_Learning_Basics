import pandas as pd 
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Mergeing train and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

target = train_df['final_status']
train_df.drop(['backers_count','final_status'],1,inplace = True)

df = pd.concat([train_df,test_df])

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

print(kickdesc[0:10])

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

# print(combine.head(10))
# print(combine.dtypes)
# print(combine.shape)
# print(df.shape)
# # print(df.head(5))
# combine.reset_index(drop=True,inplace=True)
# print(combine.head(10))

# df = df.loc[~df.index.duplicated(keep='first')]
# combine = combine.loc[~combine.index.duplicated(keep='first')]
# df = pd.concat([combine,df],1)
# print(df.dtypes)
# print(df.head(10))
# print(combine.columns)

df = pd.concat([df, combine], axis=1, join_axes=[df.index])
print(df.head(10))


