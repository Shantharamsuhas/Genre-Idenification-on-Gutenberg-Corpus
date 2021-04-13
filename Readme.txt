Run Statements for Genre prediction Program

Sample Run Statement with 2 Arguments,1.Path to stored model variables(RFC.sav,scaler.sav,label.sav),2.Path to file which needs to be genre identified.
This will ouput the predicted genre.
!python 'ATIML.py' '/content/gdrive/My Drive/ATIML/Renamed' '/content/gdrive/My Drive/ATIML/Renamed/2.html'


Variables in the main and samples

#path to input text files
textpath='/content/gdrive/My Drive/ATIML/Renamed'

#path to stored model variables
variablepath='/content/gdrive/My Drive/ATIML/Renamed'

#path to MasterID File with mapped genres
MasterID='/content/gdrive/My Drive/ATIML/MasterID.csv'

#path to features.csv
featurepath='/content/gdrive/My Drive/ATIML/Renamed/features.csv'

#path to file which needs to be genre identified
newtextpath='/content/gdrive/My Drive/ATIML/Renamed/2.html'



Functions in the main and samples

#Extract Features of the text with inputs as path to the text files and MasterID File with mapped genres
df=process_text(textpath,MasterID)

#Exploratory Visualization 
EDAVisualization(df)

#Preprocess the data with inputs as Preprocessed data, path to features.csv and #path to store model variables
X_train, X_test, y_train, y_test=preprocess(featurepath,variablepath)

#run gridsearch with inputs as Preprocessed data
best_grid=gridsearch(X_train, y_train)

#Train the model with inputs as Preprocessed data and path to store model variables
model=train_model(X_train, X_test, y_train, y_test,variablepath)

#Predict the genre with inputs as path to store model variables and path to file which needs to be genre identified
genre=predict_genre(variablepath,newtextpath)