# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup

import gzip
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
 

# %% [markdown]
# ## Read Data

# %% [markdown]
# #### For preparing the data, I am using the gzip package to open and read the dataset. The datasets consists of 15 columns of which I am extracting the 'review_body' and 'rating' column for this assignment. The dataframe 'df_review_rating' holds these extracted columns. I am converting the 'ratings' to a standard format which I am then using to create our binary classes. For simplicity, I have created a copy of the 'df_review_rating' called 'binary_df' which has an extra column called as 'classes'. This columns holds the labels for our dataset. Finally, I extracted 50000 reviews randomly from each class and stored it in the 'dataset_df' dataset

# %%
dataset_path = 'amazon_reviews_us_Office_Products_v1_00.tsv.gz'
with gzip.open(dataset_path, 'rt', encoding='utf-8') as file:
    df = pd.read_csv(file, sep='\t', on_bad_lines='skip', low_memory=False)

# %%
print(df.columns.values)

# %%
df.head(5)

# %% [markdown]
# ## Keep Reviews and Ratings

# %%
df_review_rating = df[['star_rating','review_body']]
df_review_rating

# %%
df_review_rating['star_rating']=pd.to_numeric(df_review_rating['star_rating'], errors='coerce')
df_review_rating = df_review_rating[pd.notna(df_review_rating['star_rating'])]

df_review_rating.head(5)

# %% [markdown]
#  ## We form two classes and select 50000 reviews randomly from each class.
# 
# 

# %%
df_review_rating['star_rating'].unique()

# %%
binary_df = df_review_rating.copy()

def category(row):
    if row['star_rating'] == 1 or row['star_rating'] == '1' or row['star_rating'] == 2 or row['star_rating'] == '2' or row['star_rating'] == 3 or row['star_rating'] == '3':
        return 1
    
    else:
        return 2

# %%
binary_df['class'] = df.apply(lambda row: category(row), axis=1)
binary_df

# %%
binary_df['class'].value_counts()

# %%
binary_df['class'].unique()

# %%
# Reading rows belonging to classes 1 and 2
class1_df = binary_df[binary_df['class'] == 1]
class2_df = binary_df[binary_df['class'] == 2]

# Randomly choosing 50,000 reviews of each class
random_class1_df = class1_df.sample(n = 50000, random_state=42)
random_class2_df = class2_df.sample(n = 50000, random_state=42)

# Combining the two classes to create a single dataset
dataset_df = pd.concat([random_class1_df, random_class2_df])

# Reset the indexes
dataset_df.reset_index(drop=True, inplace=True)

dataset_df

# %%
dataset_df['class'].astype(int)
dataset_df['class'].value_counts()

# %% [markdown]
# # Data Cleaning
# 
# 

# %% [markdown]
# #### The following tasks were performed for cleaning the dataset - 
# #### 1) Firstly, I looked for rows with missed values and replaced it with an empty string.
# #### 2) Converted all the reviews to lowercase using the lower() function.
# #### 3) Removed punctuations from the review by using the string.punctuation package.
# #### 4) Removed any kind of non-alphabetical characters from the reviews by tokenizing the words and checking if each character is between A-Z or a-z.
# #### 5) Removed all HTML tags and URLs from the reviews.
# #### 6) Removed any use of emojis in the reviews.
# #### 7) Finally, removed all the extra spaces from the reviews.
# 
# #### I performed contractions on the reviews as well. However, I got slightly better results when contractions was avoided.
# 

# %% [markdown]
# # Pre-processing

# %% [markdown]
# ### Removing empty reviews

# %%
print(dataset_df.isnull().values.any())
print(dataset_df.isnull().sum())

dataset_df = dataset_df.fillna('')

# %% [markdown]
# ### Storing average length of the reviews in terms of character length in your dataset before cleaning

# %%

reviewLen = pd.DataFrame()
reviewLen['before'] = dataset_df['review_body'].str.len()
print(reviewLen.head(5))

# %% [markdown]
# ### Converting reviews into lowercase

# %%
dataset_df['review_body'] = dataset_df['review_body'].str.lower()
reviewLen['lowercase'] = dataset_df['review_body'].str.len()
print(dataset_df.head(5))

# %% [markdown]
# ### Remove Punctuations

# %%
def remove_punctuations(text):
    if isinstance(text, str): 
        return ''.join(char for char in text if char not in string.punctuation)
    else:
        return text

dataset_df['review_body'] = dataset_df['review_body'].apply(remove_punctuations)
reviewLen['punctations'] = dataset_df['review_body'].str.len()

print(dataset_df.head(5))

# %% [markdown]
# ### Remove non-alphabetical characters

# %%
def remove_non_alphabetical(text):
    if isinstance(text, str): 
        return re.sub(r'[^a-zA-Z]', ' ', text)
    else:
        return text

dataset_df['review_body'] = dataset_df['review_body'].apply(remove_non_alphabetical)
reviewLen['non_alphanum'] = dataset_df['review_body'].str.len()

print(dataset_df.head(5))

# %% [markdown]
# ### Remove HTML and URLs from the reviews

# %%
dataset_df['review_body'] = dataset_df['review_body'].astype(str)

def remove_html_tags(text):
    try:
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text
    except TypeError:
        return text

def remove_urls(text):
    try:
        clean_text = re.sub(r'http\S+', '', text)
        return clean_text
    except TypeError:
        return text

dataset_df['review_body'] = dataset_df['review_body'].apply(remove_html_tags)
dataset_df['review_body'] = dataset_df['review_body'].apply(remove_urls)

reviewLen['HTML_URLs'] = dataset_df['review_body'].str.len()

print(dataset_df.head(5))

# %% [markdown]
# ### Remove Emojis

# %%
def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)

dataset_df['review_body'] = dataset_df['review_body'].apply(remove_emojis)
reviewLen['Emojis'] = dataset_df['review_body'].str.len()

print(dataset_df.head(5))

# %% [markdown]
# ### Remove extra spaces

# %%
dataset_df['review_body'] = dataset_df['review_body'].str.replace(r'\s+', ' ', regex=False)
reviewLen['extraSpaces'] = dataset_df['review_body'].str.len()

print(dataset_df.head(5))

# %% [markdown]
# ### Contractions on the reviews

# %%
# def expand_contractions(text):
#     return contractions.fix(text)

# dataset_df['review_body'] = dataset_df['review_body'].apply(expand_contractions)
# reviewLen['contractions'] = dataset_df['review_body'].str.len()

# print(dataset_df.head(5))

# %% [markdown]
# ### Average length of the reviews in terms of character length in your dataset before and after cleaning

# %%
reviewLen['after'] = dataset_df['review_body'].str.len()

print("The average length of the reviews before clearning - ",reviewLen['before'].mean())
print("The average length of the reviews after cleaning - ", reviewLen['after'].mean())

# %% [markdown]
# #### Using NLKT package to remove stopwords from the reviews and perform lemmatization on it.

# %% [markdown]
# ## Remove the stop words 

# %%
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
dataset_df['review_body'] = dataset_df['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


print(dataset_df.head(5))

# %% [markdown]
# ## Perform lemmatization  

# %%
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = nltk.word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

dataset_df['review_body'] = dataset_df['review_body'].apply(lemmatize_text)
reviewLen['lemmatization'] = dataset_df['review_body'].str.len()

print(dataset_df.head(5))


# %% [markdown]
# ### Average length of the reviews in terms of character length in your dataset before and after preprocessing

# %%
print("The average length of the reviews after cleaning - ", reviewLen['after'].mean())
print("The average length of the reviews in terms of character length after preprocessing - ", reviewLen['lemmatization'].mean())

# %% [markdown]
# #### Using sklearn to divide our dataset into train and test and finally perform TF-IDF and BOW feature extraction on the training and testing datasets.

# %% [markdown]
# # TF-IDF and BoW Feature Extraction

# %% [markdown]
# ### Dataset Splitting

# %%
Xtrain, Xtest, Ytrain, Ytest = train_test_split(dataset_df['review_body'], dataset_df['class'], test_size=0.2, random_state=42)

print("X_train shape:", Xtrain.shape)
print("X_test shape:", Xtest.shape)
print("y_train shape:", Ytrain.shape)
print("y_test shape:", Ytest.shape)

# %% [markdown]
# ### TF-IDF

# %%
tf_idf = TfidfVectorizer()
Xtrain_tf_idf = tf_idf.fit_transform(Xtrain)
Xtest_tf_idf = tf_idf.transform(Xtest)

# %% [markdown]
# ### BOW

# %%
bowVectorizer = CountVectorizer()
Xtrain_BOW = bowVectorizer.fit_transform(Xtrain)
Xtest_BOW = bowVectorizer.transform(Xtest)

# %% [markdown]
# ### Generate Scores

# %%
def get_stats(Ytest, pred):
    precision = precision_score(Ytest, pred)
    recall = recall_score(Ytest, pred)
    f1 = f1_score(Ytest, pred)

    return precision, recall, f1

# %% [markdown]
# #### Although not included, I have made use of GridSearchCV to determine the best parameters for training our models. 

# %% [markdown]
# # Perceptron Using Both Features

# %% [markdown]
# ### TF-IDF

# %%
# Initialize the Perceptron model
model_perceptron = Perceptron(alpha=0.001, max_iter=1000)

model_perceptron = model_perceptron.fit(Xtrain_tf_idf, Ytrain)
predPerceptron = model_perceptron.predict(Xtest_tf_idf)

precision_tfidf, recall_tfidf, f1_tfidf = get_stats(Ytest, predPerceptron)

# %% [markdown]
# ### BOW

# %%
# Initialize the Perceptron model
model_perceptron = Perceptron(tol=1e-03)

model_perceptron = model_perceptron.fit(Xtrain_BOW, Ytrain)
predPerceptron = model_perceptron.predict(Xtest_BOW)

precision_bow, recall_bow, f1_bow = get_stats(Ytest, predPerceptron)

# %%
print(f"TF-IDF FOR PERCEPTRON - {precision_tfidf}, {recall_tfidf}, {f1_tfidf}")
print(f"BOW FOR PERCEPTRON - {precision_bow}, {recall_bow}, {f1_bow}")

# %% [markdown]
# # SVM Using Both Features

# %% [markdown]
# ### TF-IDF

# %%
model_svm = LinearSVC(C=0.35,
    tol=0.001,
    max_iter=1000,                 #Total iterations
    random_state=16,                #Control the random number generation to control the shuffling
    penalty='l1',                  #Norm of Penalty 
    class_weight="balanced",       #Provides the weight to each class
    loss='squared_hinge',          #Specifies the Loss Function
    dual=False
)

model_svm = model_svm.fit(Xtrain_tf_idf , Ytrain)
predSVM = model_svm.predict(Xtest_tf_idf)

precision_tfidf, recall_tfidf, f1_tfidf = get_stats(Ytest, predSVM)

# %% [markdown]
# ### BOW

# %%
model_svm = LinearSVC(C=0.35,
    tol=0.001,
    max_iter=1000,                 #Total iterations
    random_state=16,                #Control the random number generation to control the shuffling
    penalty='l1',                  #Norm of Penalty 
    class_weight="balanced",       #Provides the weight to each class
    loss='squared_hinge',          #Specifies the Loss Function
    dual=False
)

model_svm = model_svm.fit(Xtrain_BOW , Ytrain)
predSVM = model_svm.predict(Xtest_BOW)

precision_bow, recall_bow, f1_bow = get_stats(Ytest, predSVM)

# %%
print(f"TF-IDF FOR SVM - {precision_tfidf}, {recall_tfidf}, {f1_tfidf}")
print(f"BOW FOR SVM - {precision_bow}, {recall_bow}, {f1_bow}")

# %% [markdown]
# # Logistic Regression Using Both Features

# %% [markdown]
# ### TF-IDF

# %%
model_LR = LogisticRegression(max_iter=10000)

model_LR = model_LR.fit(Xtrain_tf_idf , Ytrain)
predLR = model_LR.predict(Xtest_tf_idf)

precision_tfidf, recall_tfidf, f1_tfidf = get_stats(Ytest, predLR)

# %% [markdown]
# ### BOW

# %%
model_LR = LogisticRegression(max_iter=10000)

model_LR = model_LR.fit(Xtrain_BOW , Ytrain)
predLR = model_LR.predict(Xtest_BOW)

precision_bow, recall_bow, f1_bow = get_stats(Ytest, predLR)

# %%
print(f"TF-IDF FOR Logistic Regression - {precision_tfidf}, {recall_tfidf}, {f1_tfidf}")
print(f"BOW FOR Logistic Regression - {precision_bow}, {recall_bow}, {f1_bow}")

# %% [markdown]
# # Naive Bayes Using Both Features

# %% [markdown]
# ### TF-IDF

# %%
model_NB = MultinomialNB(alpha=1)

model_NB = model_NB.fit(Xtrain_tf_idf , Ytrain)
predNB = model_NB.predict(Xtest_tf_idf)

precision_tfidf, recall_tfidf, f1_tfidf = get_stats(Ytest, predNB)

# %% [markdown]
# ### BOW

# %%
model_NB = MultinomialNB(alpha=1)

model_NB = model_NB.fit(Xtrain_BOW , Ytrain)
predNB = model_NB.predict(Xtest_BOW)

precision_bow, recall_bow, f1_bow = get_stats(Ytest, predNB)

# %%
print(f"TF-IDF FOR Naive Bayes - {precision_tfidf}, {recall_tfidf}, {f1_tfidf}")
print(f"BOW FOR Naive Bayes - {precision_bow}, {recall_bow}, {f1_bow}")


