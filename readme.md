# PredictingHateSpeech

This is a research project to see how well you can predict hate speech using different NLP techniques on the same twitter dataset. My data is labelled with 3 classes, hate speech, offensive, and non-hate speech non-offensive. I wanted to see if typical nlp methods could diferentiate between tweets that are just offensive, and tweets that are hate speech. My data is in this weird grey zone of size where it seems like both tfidf based classifiers and neural net based LSTMS could be useful. I decided to compare the two methods.

## Installing 

The requirements.txt file lists the software versions I used.

## Directories

This repository consists of 4 directories:

- data
- images
- src
- notebooks

### Data

Data contains two files, which are both the same data except one is a csv and one is pickled. The data is a list of tweets each labelled with which of 3 classes it belongs too. The 0 class represents hate speech, the 1 class represents offensive text, and the 2 class represents non-offensive non hate speech tweets.

The data was pulled from another paper by Davidson et al. Here is a link to that repo: https://github.com/t-davidson/hate-speech-and-offensive-language

### Images

This directory just contains some images created during the EDA step of this project.

### Src

Src contains 3 files

- lstm_cleaner.py 

- make_model_lstm.py

- tfidfcleaner.py 

lstm_cleaner fits and transforms text to an integer sequence for use with an LSTM.

make_model_lstm builds and fits our newly cleaned data to an LSTM.

tfidfcleaner.py fits and transforms our text data to tfidf based features. 

### Notebooks

Notebooks contains 3 notebooks. LSTMExample.ipynb shows how to run and use the lstm model. TFIDFExample.ipynb shows how to run and use the tfidf model. And EDA just shows how the images were generated.

## License
[MIT](https://choosealicense.com/licenses/mit/)
