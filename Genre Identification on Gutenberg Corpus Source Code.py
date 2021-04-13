#from google.colab import drive
import pickle
import sys
import os
import math   
import pandas as pd
import numpy as np    
from numpy import mean
import argparse
import random

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
import nltk.tag.stanford as st
	
import shutil
import time
import warnings
import gender_guesser.detector as gender
	
import matplotlib.pyplot as plt
from matplotlib import pyplot

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from collections import Counter
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



def renamefile(path,MasterID):
#Renaming files wrt newID
  dfcsv = pd.read_csv(MasterID, encoding = "ISO-8859-1", engine='python')
  books_list = dfcsv.values.tolist()
  for i in range(len(dfcsv)):
    old=path+str(str(books_list[i][1]).split('.epub')[0])+'-content.html'
    new=path+str(i)+'.html'
    shutil.move(old, new)
    
#Breaks the given book into smaller parts called as chunks 
def chunking(tokenlist):
  chunkcount=math.ceil(len(tokenlist)/10000)
  chunk=[]
  max = 9999
  min = 0

  for i in range (chunkcount-1):
    x=tokenlist[min:max+1]
    chunk.append(' '.join(x))
    min=min+10000
    max=max+10000

  if min < len(tokenlist):
    x=tokenlist[min:len(tokenlist)]
    last = ' '.join(x)
    y=' '.join(tokenlist[0:10000-len(x)])
    last=last + ' ' + y
    chunk.append(last)

  return chunk

#Helps to determine whether the presented chunk has a positve sentiment , negative sentiment or neutral sentiment 
def sentimentanalysis(chunklist):
  sid = SentimentIntensityAnalyzer()
  scoreneg=0
  scorepos=0
  scoreneu=0
  # Calling the polarity_scores method on sid and passing in the text outputs a dictionary with negative, neutral, positive, and compound scores for the input text
  for i in range(len(chunklist)):
    text=' '.join(chunklist[i].split()[0:1000])
    scores = sid.polarity_scores(text)
    scoreneg=scoreneg+scores['neg']
    scorepos=scorepos+scores['pos']
    scoreneu=scoreneu+scores['neu']
  scoreneg=scoreneg/len(chunklist)
  scorepos=scorepos/len(chunklist)
  scoreneu=scoreneu/len(chunklist)  
  
  return scorepos,scoreneg,scoreneu

#Determines how difficult a passage is to understand using measures like word length , sentence length etc 
def FleschReadabilityEase(text):
	if len(text) > 0:
		return 206.835 - (1.015 * len(text.split()) / len(text.split('.')) ) - 84.6 * (sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","Y"] else 0,text))) / len(text.split()))

#Determines the number of paragraphs present in the given book
def paragraph_count(text):
	if len(text) > 0:
		return text.count('<p>')

#Helps to determine the part of speech of the word  
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(chunklist):
  lemmalist=[]
  lemmatizer = WordNetLemmatizer() 
  for i in range(len(chunklist)):
    lemmalist.append(' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(chunklist[i])]))
  return lemmalist


#Removes most frequent words of English literature like "the","etc" and so on 
def stopwordremove(tokenlist,Characters=[]):
  stop_words=set(stopwords.words("english"))
  stop_words=list(stop_words)
  stop_words=stop_words+Characters
  filtered_text=[]
  for w in tokenlist:
    if w not in stop_words:
        filtered_text.append(w.lower())
  return filtered_text

#Gives all the characters introduced in the book
def Charactertagger(text):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    st = StanfordNERTagger('/content/gdrive/My Drive/ATIML/stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
              '/content/gdrive/My Drive/ATIML/stanford-ner-4.0.0/stanford-ner.jar',
              encoding='utf-8')
    #tokenized_text = word_tokenize(text)
    tagged = st.tag(text)
  characters=[]
  for word in tagged:
    if word[1] == 'PERSON':
      characters.append(word[0].lower()) 
  characters = list(dict.fromkeys(characters))  
  return characters

#Gives normalized value of total unique words present in the chunk
def TTR(chunklist):
  score=0
  for i in range(len(chunklist)):
    tokens=nltk.word_tokenize(chunklist[i])
    types=nltk.Counter(tokens)
    score= score+(len(types)/len(tokens))*100
  score=score/len(chunklist)
  return score

#Identifies the gender of the character present in book i.e Male or Female
def genderidentify(Characters):
  scorem=0
  scoref=0

  d = gender.Detector(case_sensitive=False)
  for i in range(len(Characters)):
    sex=d.get_gender(Characters[i])
    if sex == 'male' or sex == 'mostly_male':
      scorem = scorem+1
    if sex == 'female' or sex == 'mostly_female':
      scoref = scoref+1
  return scorem,scoref

#Helps to determine the writing style of the author 
def writing_style(chunklist):
  tensefuture,  tensepast,  tensepresent,  Coordinatingconjunction,  PersonalPronoun,  PossessivePronoun=0,0,0,0,0,0
  for i in range(len(chunklist)):
    tokens=nltk.word_tokenize(chunklist[i])
    tagged = nltk.pos_tag(tokens)
    for word in tagged:
      if word[1] in ["VBC", "VBF","MD"]:
        tensefuture =tensefuture+1
      if word[1] in ["VBD", "VBN"]:
        tensepast =tensepast+ 1
      if word[1] in ["VBG", "VBP","VBZ"]:
        tensepresent =tensepresent+ 1
      if word[1] in ["CC"]:
        Coordinatingconjunction =Coordinatingconjunction+ 1
      if word[1] in ["PRP"]:
        PersonalPronoun =PersonalPronoun+ 1
      if word[1] in ["PRP$"]:
        PossessivePronoun =PossessivePronoun+ 1
  
  Coordinatingconjunction=Coordinatingconjunction/len(chunklist)
  tensefuture=tensefuture/len(chunklist)
  tensepast=tensepast/len(chunklist)
  tensepresent=tensepresent/len(chunklist)
  PersonalPronoun=PersonalPronoun/len(chunklist)
  PossessivePronoun=PossessivePronoun/len(chunklist)
  return tensefuture,  tensepast,  tensepresent,  Coordinatingconjunction,  PersonalPronoun,  PossessivePronoun

#Outputs the condensed feature representation of the provided dataset
def feature_extractor(text):
  feature=[]
  paragraphCount=paragraph_count(text)
  text=text.replace('<p>','')
  Readability=FleschReadabilityEase(text)
  tokenlist=word_tokenize(text)
  
  j=0
  for i in range(len(tokenlist)):
    if tokenlist[j] == '``':
      tokenlist.remove('``')
      j=j-1
      i=i+1
    elif tokenlist[j] == "''":
      tokenlist.remove("''")
      j=j-1
      i=i+1

    j=j+1
  
  Characters=Charactertagger(tokenlist)
  male,female=genderidentify(Characters)
  tokenizer = nltk.RegexpTokenizer(r"\w+")
  textwithoutpunctuation = tokenizer.tokenize(text)
  stopwordremoved=stopwordremove(textwithoutpunctuation,Characters)
  chunklistTTR=chunking(stopwordremoved)
  lemmalistTTR=lemmatize(chunklistTTR)
  TypeTokenRatio=TTR(lemmalistTTR)
  chunklist=chunking(tokenlist)
  sentimentpos,sentimentneg,sentimentneu=sentimentanalysis(chunklist)
  Names=len(Characters)
  quote=(text.count('\"'))/len(chunklist)
  paragraphCount=paragraphCount/len(chunklist)
  Setting=Names+quote
  tensefuture,  tensepast,  tensepresent,  Coordinatingconjunction,  PersonalPronoun,  PossessivePronoun=writing_style(chunklist)
  feature=[paragraphCount,Readability,male,female,TypeTokenRatio,sentimentpos,sentimentneg,sentimentneu,Names,Setting,tensefuture,  tensepast,  tensepresent,  Coordinatingconjunction,  PersonalPronoun,  PossessivePronoun]
  for i in range(len(feature)):
    feature[i]=round(feature[i],5)
  return feature

def process_text(textpath,MasterID):  
  os.chdir(textpath)
  dfcsv = pd.read_csv(MasterID, encoding = "ISO-8859-1", engine='python')
  books_list = dfcsv.values.tolist()
  df = pd.DataFrame(columns=("FileID","paragraphCount","Readability","male","female","TypeTokenRatio","sentimentpos","sentimentneg","sentimentneu","Names","Setting","tensefuture",  "tensepast",  
                             "tensepresent",  "Coordinatingconjunction",  "PersonalPronoun",  "PossessivePronoun","Genre"))
  startfull = time.time()
  for i in range(len(dfcsv)):
    start = time.time()
    file = open(str(i)+".html", "r")
    text=str(file.read())
    if(len(text)>0):
      feature=feature_extractor(text)
      feature.insert(0,str(i))
      feature.insert(17,books_list[i][3])
      print(feature)
      df.loc[len(df)] = feature
      with open('tempdf', 'wb') as f:
        pickle.dump(df, f)
      end = time.time()
      print(i,end - start)
  df.to_csv('features.csv')
  endfull = time.time()
  print(endfull - startfull)
  return df

#Outputs the confusion matrix 
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    for i in range(len(cm_sum)):
      if cm_sum[i] == 0:
        cm_sum[i]=1
         
    cm_perc = cm / cm_sum.astype(float) * 100
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    #plt.savefig(filename)
    plt.show()

def EDAVisualization(df):
  
  sns.set(color_codes=True)
  x = df['Names']
  sns.pairplot(df)
  x = df['Genre']
  y=Counter(x)
  plt.bar(*zip(*y.most_common()), width=.5, color='g')
  plt.ylabel('Frequency')
  plt.xlabel('Genre')
  plt.xticks(rotation='vertical')
  plt.show()

def gridsearch(train_features, train_labels):
  # Create the parameter grid based on the results of random search 
  param_grid = {
      'bootstrap': [True],
      'max_depth': [80, 90, 100, 110,None],
      'max_features': [2, 3,'auto'],
      'min_samples_leaf': [1,2,3, 4, 5],
      'min_samples_split': [1,2,3,4,5,6,7,8, 10, 12],
      'n_estimators': [6,7,8,9,10,11,12,13,14,15,16],
      'criterion':['entropy','gini']
  }
  # Create a based model
  rf = RandomForestClassifier()
  # Instantiate the grid search model
  grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)
  # Fit the grid search to the data
  grid_search.fit(train_features, train_labels)
  grid_search.best_params_
  {'bootstrap': True,
  'max_depth': 80,
  'max_features': 3,
  'min_samples_leaf': 5,
  'min_samples_split': 12,
  'n_estimators': 100}
  best_grid = grid_search.best_estimator_
  return best_grid
def preprocess(featurepath,variablepath):
  df = pd.read_csv(featurepath, encoding = "ISO-8859-1", engine='python')
  del df['FileID']
  del df['Unnamed: 0']
  dataset=df.values
  #turning textual data to numerical
  from sklearn.preprocessing import LabelEncoder
  labelencoder_genre = LabelEncoder() #independent variable encoder
  dataset[:,16] = labelencoder_genre.fit_transform(dataset[:,16])
  #print(list(labelencoder_genre.inverse_transform([0,1,2,3,4,5,6,7,8])))
  #splitting the dataset into the source variables (independant variables) and the target variable (dependant variable)
  X = dataset[:,:-1] #all columns except the last one
  y = dataset[:,len(dataset[0])-1] #only the last column
  y=y.astype('int')
  train_size=0.8
  classcount={5: 792, 2: 111, 7: 50, 8: 40, 6: 40, 3: 30, 0: 30, 4: 30, 1: 30}
  sm = SMOTE(random_state=0,sampling_strategy=classcount,k_neighbors=1)
  X, y = sm.fit_resample(X, y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-train_size, random_state = 0,stratify=y)
  classcount={5: int(792*train_size), 2: int(777*train_size), 7: int(700*train_size), 8: int(640*train_size), 6: int(640*train_size), 3:int(625*train_size) , 0: int(510*train_size), 4: int(510*train_size), 1: int(500*train_size)}
  sm = SMOTE(random_state=0,sampling_strategy=classcount,k_neighbors=5)
  X_train, y_train = sm.fit_resample(X_train, y_train)
  # Feature Scaling
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  os.chdir(variablepath)
  labelcode=labelencoder_genre
  filename1 = 'label.sav'
  pickle.dump(labelcode, open(filename1, 'wb'))
  filename2 = 'scaler.sav'
  pickle.dump(sc, open(filename2, 'wb'))
  return X_train, X_test, y_train, y_test


def evaluate(model,X_test, y_test):
  result = model.score(X_test, y_test)
  print("Accuracy: %.3f%%" % (result*100.0))
  y_pred = model.predict(X_test)
  print("F1 Score: ", f1_score(y_test, y_pred, average="macro"))
  print("Precision Score: ", precision_score(y_test, y_pred, average="macro"))
  print("Recall Score: ", recall_score(y_test, y_pred, average="macro")) 
  print("Balanced accuracy score : ",balanced_accuracy_score(y_test, y_pred))
  cm_analysis(y_test, y_pred, model.classes_, ymap=None, figsize=(10,10))

def train_model(X_train, X_test, y_train, y_test,variablepath):
  # define pipeline
  steps = [('model', RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='entropy', max_depth=90, max_features=2,
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=5,
                        min_weight_fraction_leaf=0.0, n_estimators=16,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False))]
  pipeline = Pipeline(steps=steps)
  # evaluate pipeline
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=4, random_state=1)
  scores = cross_val_score(pipeline, X_train, y_train, scoring='f1_micro', cv=cv, n_jobs=-1)
  score = mean(scores)
  print('Cross Validated f1 score: %.3f' % score)
  pipeline.fit(X_train,y_train)
  y_pred = pipeline.predict(X_test)
  evaluate(pipeline,X_test, y_test)
  os.chdir(variablepath)
  # save the model to disk
  filename = 'RFC.sav'
  pickle.dump(pipeline, open(filename, 'wb'))
  return pipeline

def predict_genre(variablepath,filenametext):
  os.chdir(variablepath)
  filename = 'RFC.sav'
  model = pickle.load(open(filename, 'rb'))
  filename1 = 'label.sav'
  label = pickle.load(open(filename1, 'rb'))
  filename2 = 'scaler.sav'
  sc = pickle.load(open(filename2, 'rb'))
  file = open(filenametext, "r")
  text=str(file.read())
  if(len(text)>0):
    feature=[feature_extractor(text)]
    
    X_test = sc.transform(feature)
    y_pred = model.predict(X_test)
    print('Predicted Genre:',label.inverse_transform([y_pred]))
    return label.inverse_transform([y_pred])

my_parser = argparse.ArgumentParser(description='List the content of a folder')
my_parser.add_argument('VarPath',
                       metavar='vpath',
                       type=str,
                       help='the path to stored Variables')
my_parser.add_argument('TextPath',
                       metavar='tpath',
                       type=str,
                       help='the path to list')
args = my_parser.parse_args()

#path to text files
textpath='/content/gdrive/My Drive/ATIML/Renamed'

#path to stored model variables
#variablepath='/content/gdrive/My Drive/ATIML/Renamed'
variablepath = args.VarPath
if not os.path.isdir(variablepath):
    print('The Variable path specified does not exist')
    sys.exit()
#path to MasterID File with mapped genres
MasterID='/content/gdrive/My Drive/ATIML/MasterID.csv'

#path to features.csv
featurepath='/content/gdrive/My Drive/ATIML/Renamed/features.csv'

#path to file which needs to be genre identified
#newtextpath='/content/gdrive/My Drive/ATIML/Renamed/2.html'
newtextpath=args.TextPath

#Extract Features of the text with inputs as path to the text files and MasterID File with mapped genres
#df=process_text(textpath,MasterID)

#Exploratory Visualization 
#EDAVisualization(df)

#Preprocess the data with inputs as Preprocessed data, path to features.csv and #path to store model variables
#X_train, X_test, y_train, y_test=preprocess(featurepath,variablepath)

#run gridsearch with inputs as Preprocessed data
#best_grid=gridsearch(X_train, y_train)

#Train the model with inputs as Preprocessed data and path to store model variables
#model=train_model(X_train, X_test, y_train, y_test,variablepath)

#Predict the genre with inputs as path to store model variables and path to file which needs to be genre identified
genre=predict_genre(variablepath,newtextpath)