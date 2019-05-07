from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

#ListyListLists
line_store = [] 
word_pairs = []
comma_split = []
X_train = []
y_train = []
results = []
X_test = {}

#   open train file to be used as input; encoding/utf-8 handles chars with diacritics;
#   iterates through file, using a variable to store all lines with leading/trailing whitespace 
#   removed by strip; 
with open('train.txt', encoding = 'utf-8') as file:
    for line in file:
        line_store.append(line.strip())
        
#   does basically the same thing as the last one except does it for the list of word options;
#   comma_split line makes it so that the word pairs are in lists
with open('IrishWordsList.txt', encoding = 'utf-8') as file:
    for line in file:
        word_pairs.append(line.strip())
for i in word_pairs:
    comma_split.append(i.split(', '))
    
#   makes a list of 25 of empty lists, separates out the sentences with a word from the lists
for x in range(25):
    X_train.append([])
    y_train.append([])
    
#   iterates through line_store and the word pairs, then check see if either of the words in the word pair are in the line.
#   enumerate makes it so that each word in the list being passed receives an indexed number,
#   which helps identify the the list with the word that's been identified in the line.
for w in line_store:
    for  i,v in enumerate(comma_split):
        if v[0] in w.split():
            X_train[i].append(w)
            y_train[i].append(v[0])
        elif v[1] in w.split():
            X_train[i].append(w)
            y_train[i].append(v[1])

#   formatting test (Y) data so that { } are removed and words extracted
with open('test.txt', encoding = 'utf-8') as file:
    for line in file:
        lbrac = line.find('{')
        rbrac = line.rfind('}')
        key = (line[lbrac + 1: rbrac])
        value = line[:lbrac] + line[rbrac + 1:]
        if key in X_test.keys():
            X_test[key].append(value)
        else:
            X_test[key] = [value]


#https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
classy = MultinomialNB()
bagging = BaggingClassifier(classy)
vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2), min_df = 0.001)

for x in range(25):
    vectorx = vectorizer.fit_transform(X_train[x],y_train[x])
    vectory = y_train[x]
    vectort = vectorizer.transform(X_test[comma_split[x][0]+'|'+ comma_split[x][1]])
    bagging.fit(vectorx, vectory)
    results.append(bagging.predict_proba(vectort))
    print(x)

with open('submission.csv', 'w', encoding = 'utf-8') as file:
    for x in results:
        for y in x:
            file.write(str(y[0]) + '\n')
    file.close