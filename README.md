# Natural Language Processing (NLP)
Python code to demonstrate [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing). You will be able to train and test the model on the given data set.

# Input & Output
The input comes in the form of a tab separated value file with index, title, url, publication and category columns. No headers are present.
Below is a sample of the data.

![news_corpora](https://github.com/azmelcmie/nlp_csv/blob/master/img/new_corpora.PNG)

 
The results can be viewed in the form of a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) that will show how well your algorithm is performing.

![confusion_matrix](https://github.com/azmelcmie/nlp_csv/blob/master/img/confusion_matrix.PNG)


## Environment
Coded and tested in Anaconda version 4.4.0 using the Spyder 3.1.4 environment.

## Usage
Run the **nlpcsva.py** file from within the Spyder environment.
Select all code and press CTRL-ENTER to run the program.
Double-click on confusion matrix (cm) in Variable explorer to view the confusion matrix table.

To experiment with your own data, you can edit the following lines.

Edit the location and name of your dataset along with the required columns (**lines 9-11**):

````python
dataset = pd.read_csv('data/news_corpora.csv', header=None, delimiter = '\t', quoting = 3, engine='python')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
````

The Naive Bayes classification model is used in this particular case.
