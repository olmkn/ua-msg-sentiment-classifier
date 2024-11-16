# %% [markdown]
# # Analyzing Sentiment in Ukrainian Telegram Texts

# %% [markdown]
# ## Importing Dependencies

# %% [code]
# Data manipulation and analysis
import numpy as np  # Numerical operations and array manipulation
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting library
import pickle  # Serialization/deserialization of Python objects
import time  # Time-related functions

# Deep learning and neural networks
import tensorflow as tf  # Machine learning framework
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenization of text data
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Padding sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, GlobalMaxPool1D, SpatialDropout1D  # Neural network layers
from tensorflow.keras import regularizers  # Regularization techniques

# Machine learning and evaluation
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
from sklearn.preprocessing import LabelEncoder  # Label encoding
from sklearn import metrics  # Metrics for evaluating models
from sklearn.metrics import auc, classification_report, confusion_matrix  # Additional metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Text feature extraction
from sklearn.pipeline import Pipeline  # Pipeline for chaining preprocessing and modeling steps
from sklearn.model_selection import GridSearchCV  # Grid search for hyperparameter tuning

# Text processing and visualization
from wordcloud import WordCloud  # Word cloud visualization
import pymorphy2  # Morphological analysis for Russian language
import re  # Regular expressions for pattern matching in text

# Operating system functions
import os

# Kaggle-specific imports (for running notebook in a Kaggle notebook)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [markdown]
# # Loading Data

# %% [code]
df = pd.read_csv('/kaggle/input/telegram-uk-6k-shuffled/telegram_uk_6k_shuffled.csv')
df.head()

# %% [markdown]
# # Cleaning Data

# %% [code]
missing_values_count = df.isna().sum()
print(missing_values_count)

# %% [code]
df = df.dropna()

# %% [code]
df.duplicated().sum()

# %% [code]
df = df.drop_duplicates()

# %% [code]
# Converting label numbers to sentiment labels
lab_to_sentiment = {-2:"Negative", 2:"Positive"}
def label_decoder(label):
  return lab_to_sentiment[label]
df.Label = df.Label.apply(lambda x: label_decoder(x))
df.head()

# %% [code]
# Dropping unnecessary columns
df.drop(columns=['group_id','group_username', 'post_id', 'post_date', 'post_msg', 'msg_id', 'msg_date', 'msg_lang'], inplace=True)
df

# %% [code] 
val_count = df.Label.value_counts()

plt.figure(figsize=(8,4))
plt.bar(val_count.index, val_count.values)
plt.title("Sentiment Data Distribution")

# %% [markdown]
# # Preprocessing Data

# %% [code] 
df['proc_msg'] = df.msg.copy()

# %% [code] 
text_cleaning_re_uk = "@\S+|https?:\S+|http?:\S|[^А-Яа-яҐґЄєІіЇї0-9]+"

# %% [code] 
with open('/kaggle/input/ukrainian-stop-words/ukrainian', 'r', encoding='utf-8') as f:
    stop_words_uk = [line.strip() for line in f]

# %% [code]
morph = pymorphy2.MorphAnalyzer(lang='uk')

# %% [code] 
def preprocess_uk(text):
  text = re.sub(text_cleaning_re_uk, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words_uk:
        token = morph.parse(token)[0].normal_form
        tokens.append(token)
  return " ".join(tokens)

# %% [code] 
df['proc_msg'] = df['proc_msg'].apply(lambda x: preprocess_uk(x))

# %% [code] 
missing_values_count = df.isna().sum()
print(missing_values_count)

# %% [code] 
df.duplicated().sum()

# %% [code] 
df = df.drop_duplicates()

# %% [code] 
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

wc_pos = WordCloud(max_words=2000, width=800, height=400).generate(" ".join(df[df.Label == 'Positive'].proc_msg))
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].set_title('Positive Words')

wc_neg = WordCloud(max_words=2000, width=800, height=400).generate(" ".join(df[df.Label == 'Negative'].proc_msg))
axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].set_title('Negative Words')

# Hide axis
for ax in axes:
    ax.axis('off')

plt.show()

# %% [markdown]
# # Training Classificators

# %% [code] 
TRAIN_SIZE = 0.8

# %% [markdown]
# ## Classical Machine Learning Approach

# %% [code] 
ml_x_train, ml_x_test, ml_y_train, ml_y_test = train_test_split(df['proc_msg'], df['Label'], test_size=1-TRAIN_SIZE, random_state=7)


# %% [code] 
scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']

# %% [markdown]
# ### Naive Bayes (NB)

# %% [code] 
from sklearn.naive_bayes import MultinomialNB

# %% [code] 
NB_pipeline = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB())])

NB_param = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': [1, 1e-1, 1e-2]
}

# %% [code] 
NB = GridSearchCV(NB_pipeline, NB_param, cv=10, scoring=scores, refit='f1_weighted')
NB.fit(ml_x_train, ml_y_train)

# %% [code]
NB.best_params_

# %% [code]
NB.best_score_

# %% [code]
from sklearn.metrics import cohen_kappa_score

# %% [code]
MNB_predictions = NB.predict(ml_x_test)
MNB_predictions_train = NB.predict(ml_x_train)
MNB_probabilities = NB.predict_proba(ml_x_test)

MNB_classification_report_dict = classification_report(ml_y_test, MNB_predictions, output_dict=True)

MNB_accuracy_score_test = metrics.accuracy_score(MNB_predictions, ml_y_test)
MNB_accuracy_score_train = metrics.accuracy_score(MNB_predictions_train, ml_y_train)

ml_y_test_num = ml_y_test.map({'Negative': 0, 'Positive': 1})
fprMNB, tprMNB, thresholdsMNB = metrics.roc_curve(ml_y_test_num, MNB_probabilities[:, 1])
AUC_MNB = auc(fprMNB, tprMNB)



print("Multinominal Naive Bayes Classifier\n")
print(f"Accuracy Score (Test): {MNB_accuracy_score_test*100:.2f}% \nAccuracy Score (Train): {MNB_accuracy_score_train*100:.2f}% \nAUC: {AUC_MNB*100:.2f}%\n")


MNB_cohen = cohen_kappa_score(MNB_predictions, ml_y_test)
print(f"Cohen Kappa Score: {MNB_cohen}")
print(print_classification_report(MNB_classification_report_dict))

# %% [code] 
plt.figure(figsize=(6,6))
plot_confusion_matrix(metrics.confusion_matrix(y_pred=MNB_predictions,y_true=ml_y_test), classes=df.Label.unique(), title="Матриця невідповідностей\n(наївний баєсів класифікатор)")
plt.show()

# %% [code] 
plot_classification_report(classification_report(ml_y_test, MNB_predictions), title='Наївний Баєсів Класифікатор')

# %% [markdown]
# ### Support Vector Machine (SVM)

# %% [code] 
from sklearn import svm

# %% [code] 
SVM_pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', svm.SVC())])

SVM_param = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
            'tfidf__use_idf': (True, False),
            'tfidf__norm': ('l1', 'l2'),
            'clf__kernel': ['linear'],
            'clf__gamma':['auto'],
            'clf__probability': [True]
}

# %% [code] 
SVM = GridSearchCV(SVM_pipeline, SVM_param, cv=10, scoring=scores, refit='f1_weighted')
SVM.fit(ml_x_train, ml_y_train)

# %% [code] 
SVM.best_params_

# %% [code] 
SVM.best_score_

# %% [code] 
SVM_predictions = SVM.predict(ml_x_test)
SVM_predictions_train = SVM.predict(ml_x_train)
SVM_probabilities = SVM.predict_proba(ml_x_test)

SVM_classification_report_dict = classification_report(ml_y_test, SVM_predictions, output_dict=True)

SVM_accuracy_score_test = metrics.accuracy_score(SVM_predictions, ml_y_test)
SVM_accuracy_score_train = metrics.accuracy_score(SVM_predictions_train, ml_y_train)

ml_y_test_num = ml_y_test.map({'Negative': 0, 'Positive': 1})
fprSVM, tprSVM, thresholdsSVM = metrics.roc_curve(ml_y_test_num, SVM_probabilities[:, 1])
SVM_AUC = auc(fprSVM, tprSVM)

print("Support Vector Machine Classifier\n")
print(f"Accuracy Score (Test): {SVM_accuracy_score_test*100:.2f}% \nAccuracy Score (Train): {MNB_accuracy_score_train*100:.2f}% \nAUC: {SVM_AUC*100:.2f}%\n")

SVM_cohen = cohen_kappa_score(SVM_predictions, ml_y_test)
print(f"Cohen Kappa Score: {SVM_cohen}")
print(print_classification_report(SVM_classification_report_dict))

# %% [code] 
plt.figure(figsize=(6,6))
plot_confusion_matrix(metrics.confusion_matrix(y_pred=SVM_predictions,y_true=ml_y_test), classes=df.Label.unique(), title="Матриця невідповідностей\n(метод опорних векторів)")
plt.show()

# %% [code] 
plot_classification_report(classification_report(ml_y_test, SVM_predictions), title='Метод опорних векторів')

# %% [markdown]
# ### AdaBoost

# %% [code] 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# %% [code] 
AdaBoost_pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', AdaBoostClassifier())])

AdaBoost_param = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
                  'tfidf__use_idf': (True, False),
                  'tfidf__norm': ['l1', 'l2'],
                  'clf__n_estimators': [10, 50, 100, 500],
                  'clf__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
                  'clf__estimator': [DecisionTreeClassifier()]
}

# %% [code] 
AdaBoost = GridSearchCV(AdaBoost_pipeline, AdaBoost_param, cv=10, scoring=scores, refit='f1_weighted')
AdaBoost

# %% [code] 
AdaBoost.fit(ml_x_train, ml_y_train)

# %% [code] 
AdaBoost.best_params_

# %% [code] 
AdaBoost.best_score_

# %% [code] 
AdaBoost_predictions = AdaBoost.predict(ml_x_test)
AdaBoost_predictions_train = AdaBoost.predict(ml_x_train)
AdaBoost_probabilities = AdaBoost.predict_proba(ml_x_test)

AdaBoost_classification_report_dict = classification_report(ml_y_test, AdaBoost_predictions, output_dict=True)


AdaBoost_accuracy_score_test = metrics.accuracy_score(AdaBoost_predictions, ml_y_test)
AdaBoost_accuracy_score_train = metrics.accuracy_score(AdaBoost_predictions_train, ml_y_train)

ml_y_test_num = ml_y_test.map({'Negative': 0, 'Positive': 1})
fprAdaBoost, tprAdaBoost, thresholdsAdaBoost = metrics.roc_curve(ml_y_test_num, AdaBoost_probabilities[:, 1])
AdaBoost_AUC = auc(fprAdaBoost, tprAdaBoost)

print("Adaptive Boosting Classifier\n")
print(f"Accuracy Score (Test): {AdaBoost_accuracy_score_test*100:.2f}% \nAccuracy Score (Train): {AdaBoost_accuracy_score_train*100:.2f}% \nAUC: {AdaBoost_AUC*100:.2f}%\n")

AdaBoost_cohen = cohen_kappa_score(AdaBoost_predictions, ml_y_test)
print(f"Cohen Kappa Score: {AdaBoost_cohen}")
print(print_classification_report(AdaBoost_classification_report_dict))

# %% [code] 
plt.figure(figsize=(6,6))
plot_confusion_matrix(metrics.confusion_matrix(y_pred=AdaBoost_predictions,y_true=ml_y_test), classes=df.Label.unique(), title="Матриця невідповідностей\n(адаптивне підсилення)")
plt.show()

# %% [code] 
plot_classification_report(classification_report(ml_y_test, AdaBoost_predictions), title='Адаптивне підсилення')

# %% [markdown]
# ### Random Forest (RF)

# %% [code] 
from sklearn.ensemble import RandomForestClassifier

# %% [code] 
RF_pipeline = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier())])

RF_param = {'vect__ngram_range': [(1, 1)],
            'tfidf__use_idf': [True],
            'tfidf__norm': ['l1'],
            'clf__n_estimators': [50, 100, 200, 400],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
}

# %% [code] 
RF = GridSearchCV(RF_pipeline, RF_param, cv=10, scoring=scores, refit='f1_weighted')
RF

# %% [code] 
RF.fit(ml_x_train, ml_y_train)

# %% [code] 
RF.best_params_

# %% [code] 
RF.best_score_

# %% [code] 
RF_predictions = RF.predict(ml_x_test)
RF_predictions_train = RF.predict(ml_x_train)
RF_probabilities = RF.predict_proba(ml_x_test)

RF_classification_report_dict = classification_report(ml_y_test, RF_predictions, output_dict=True)

RF_accuracy_score_test = metrics.accuracy_score(RF_predictions, ml_y_test)
RF_accuracy_score_train = metrics.accuracy_score(RF_predictions_train, ml_y_train)

ml_y_test_num = ml_y_test.map({'Negative': 0, 'Positive': 1})
fprRF, tprRF, thresholdsRF = metrics.roc_curve(ml_y_test_num, RF_probabilities[:, 1])
RF_AUC = auc(fprRF, tprRF)

print("Random Forest Classifier\n")
print(f"Accuracy Score (Test): {RF_accuracy_score_test*100:.2f}% \nAccuracy Score (Train): {RF_accuracy_score_train*100:.2f}% \nAUC: {RF_AUC*100:.2f}%\n")


RF_cohen = cohen_kappa_score(RF_predictions, ml_y_test)
print(f"Cohen Kappa Score: {RF_cohen}")
print(print_classification_report(RF_classification_report_dict))

# %% [code] 
plt.figure(figsize=(6,6))
plot_confusion_matrix(metrics.confusion_matrix(y_pred=RF_predictions,y_true=ml_y_test), classes=df.Label.unique(), title="Матриця невідповідностей\n(метод випадкового лісу)")
plt.show()

# %% [code] 
plot_classification_report(classification_report(ml_y_test, RF_predictions), title='Метод випадкового лісу')


# %% [markdown]
# ### Save Best Models

# %% [code]

with open('MNB.pkl', 'wb') as file:
    pickle.dump(NB, file)

with open('SVM.pkl', 'wb') as file:
    pickle.dump(SVM, file)
    
with open('AdaBoost.pkl', 'wb') as file:
    pickle.dump(AdaBoost, file)
    
with open('RF.pkl', 'wb') as file:
    pickle.dump(RF, file)

# %% [markdown]
# ## Deep Learning Approach

# %% [markdown]
# ### Utiels

# %% [code] 
def plot_learning_curve(model_hist):
    s, (at, al) = plt.subplots(2,1)
    at.plot(model_hist.history['accuracy'], c= 'b')
    at.plot(model_hist.history['val_accuracy'], c='r')
    at.set_title('model accuracy')
    at.set_ylabel('accuracy')
    at.set_xlabel('epoch')
    at.legend(['train', 'val'], loc='upper left')

    al.plot(model_hist.history['loss'], c='m')
    al.plot(model_hist.history['val_loss'], c='c')
    al.set_title('model loss')
    al.set_ylabel('loss')
    al.set_xlabel('epoch')
    al.legend(['train', 'val'], loc = 'upper left')

# %% [code] 
def decode_sentiment(score):
    return "Positive" if score>0.5 else "Negative"

# %% [markdown]
# ### Data Prepiration

# %% [code] 
df['word_count'] = df['proc_msg'].apply(lambda x: len(str(x).split()))

avg_word_count = df['word_count'].mean()

plt.figure(figsize=(8, 6))
plt.hist(df['word_count'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(avg_word_count, color='red', linestyle='dashed', linewidth=1)
plt.text(avg_word_count + 0.5, 100, f'Avg. Word Count: {avg_word_count:.2f}', color='red')
plt.title('Distribution of Word Count')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

df.drop(columns=['word_count'], inplace=True)

# %% [code] 
MAX_SEQUENCE_LENGTH = 20

# %% [code] 
dl_train_data, dl_test_data = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=7) 
print("Train Data size:", len(dl_train_data))
print("Test Data size", len(dl_test_data))

# %% [code] 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dl_train_data.proc_msg)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)

# %% [code] 
dl_x_train = pad_sequences(tokenizer.texts_to_sequences(dl_train_data.proc_msg),
                        maxlen = MAX_SEQUENCE_LENGTH)
dl_x_test = pad_sequences(tokenizer.texts_to_sequences(dl_test_data.proc_msg),
                       maxlen = MAX_SEQUENCE_LENGTH)

print("Training X Shape:", dl_x_train.shape)
print("Testing X Shape:", dl_x_test.shape)

# %% [code] 
labels = dl_train_data.Label.unique().tolist()

# %% [code] 
encoder = LabelEncoder()
encoder.fit(dl_train_data.Label.to_list())

dl_y_train = encoder.transform(dl_train_data.Label.to_list())
dl_y_test = encoder.transform(dl_test_data.Label.to_list())

dl_y_train = dl_y_train.reshape(-1,1)
dl_y_test = dl_y_test.reshape(-1,1)

print("y_train shape:", dl_y_train.shape)
print("y_test shape:", dl_y_test.shape)

# %% [code] 
EMB = '/kaggle/input/ubercorpus-lowercased-lemmatized-300d-p/ubercorpus.lowercased.lemmatized.300d.txt'
EMBEDDING_DIM = 300

# %% [code] 
embeddings_index = {}

f = open(EMB)
for line in f:
  values = line.split()
  word = value = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' %len(embeddings_index))

# %% [code] 
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

# %% [code] 
embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False)

# %% [markdown]
# ### LSTM

# %% [code] 
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

# %% [code] 
# Model 1

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
lstm_x = LSTM(64)(embedding_sequences)
outputs = Dense(1, activation='sigmoid')(lstm_x)

lstm = tf.keras.Model(sequence_input, outputs)

print(lstm.summary())

# %% [code] 
# Model 2

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
lstm_x = SpatialDropout1D(0.2)(embedding_sequences)
lstm_x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(lstm_x)
outputs = Dense(1, activation='sigmoid')(lstm_x)

lstm = tf.keras.Model(sequence_input, outputs)

print(lstm.summary())

# %% [code] 
# Model 3

equence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
lstm_x = LSTM(128, return_sequences=True)(embedding_sequences)
lstm_x = SpatialDropout1D(0.5)(lstm_x)
lstm_x = LSTM(128)(lstm_x)
lstm_x = Dense(64, activation='relu', 
                      kernel_regularizer=regularizers.l2(0.01), 
                      activity_regularizer=regularizers.l1(0.01))(lstm_x)
outputs = Dense(1, activation='sigmoid')(lstm_x)

lstm = tf.keras.Model(sequence_input, outputs)

print(lstm.summary())

# %% [code] 
# Model 4

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
lstm_x = LSTM(50, return_sequences = True)(embedding_sequences)
lstm_x = GlobalMaxPool1D()(lstm_x)
lstm_x = Dense(16, activation="relu")(lstm_x)
lstm_x = Dropout(0.8)(lstm_x)
outputs = Dense(1, activation="sigmoid")(lstm_x)

lstm = tf.keras.Model(sequence_input, outputs)

print(lstm.summary())

# %% [code] 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

optimizer = Adam(learning_rate=LR)
lstm.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,
                                     min_lr = 0.01,
                                     monitor = 'val_loss',
                                     verbose = 1)

# %% [code] 
lstm_hist = lstm.fit(dl_x_train, dl_y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dl_x_test, dl_y_test), callbacks=[ReduceLROnPlateau])

# %% [code] 
plot_learning_curve(lstm_hist)

# %% [code] 
lstm_scores = lstm.predict(dl_x_test, verbose=1)
lstm_predictions = [decode_sentiment(lstm_score) for lstm_score in lstm_scores]

# %% [code] 
LSTM_classification_report_dict = classification_report(list(dl_test_data.Label), lstm_predictions, output_dict=True)

LSTM_accuracy_score = metrics.accuracy_score(lstm_predictions, ml_y_test)

fprLSTM, tprLSTM, thresholdsLSTM = metrics.roc_curve(dl_y_test, lstm_scores)
LSTM_AUC= auc(fprLSTM, tprLSTM)

print(f"Accuracy Score: {LSTM_accuracy_score*100:.2f}% \nAUC: {LSTM_AUC*100:.2f}%\n")

LSTM_cohen = cohen_kappa_score(lstm_predictions, ml_y_test)
print(f"Cohen Kappa Score: {LSTM_cohen}")
print(print_classification_report(LSTM_classification_report_dict))

# %% [code] 
plot_classification_report(classification_report(list(dl_test_data.Label), lstm_predictions), title='LSTM')

# %% [code] 
lstm_cnf_matrix = confusion_matrix(dl_test_data.Label.to_list(), lstm_predictions)
plt.figure(figsize=(6,6))
plot_confusion_matrix(lstm_cnf_matrix, classes=dl_test_data.Label.unique(), title="Матриця невідповідностей")
plt.show()