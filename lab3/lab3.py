import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        #print(self.pima.head())
        self.X_test = None
        self.y_test = None
        

    def define_feature(self, cols):
        #print('Feature selected: ', cols)
    
        X = self.pima[cols]
        y = self.pima.label
        return X, y
    
    def train(self, cols):
        # split X and y into training and testing sets
        X, y = self.define_feature(cols)
        #scaler = MinMaxScaler()
        #scaler.fit(X)
        #X = scaler.transform(X)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        #print(['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'])
        #print(logreg.coef_)
        return logreg
    
    def predict(self, cols):
        model = self.train(cols)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)

    def bucketize_a_feature(self, col):
        X = self.pima[col]
        est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
        est.fit(X.values.reshape(-1, 1))
        bc_X = est.transform(X.values.reshape(-1,1))
        self.pima[col] = bc_X
       


    
if __name__ == "__main__":
    #classifer = DiabetesClassifier()
    #result = classifer.predict( ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age'])






    cols = [['pregnant', 'insulin', 'bmi', 'age'], ['insulin','glucose', 'pregnant','bmi', 'pedigree','age'], ['pregnant', 'bmi', 'glucose', 'bp']]

    #classifer = DiabetesClassifier()
    #result = classifer.predict(cols[2])
    #print(f"Predicition={result}")
    #score = classifer.calculate_accuracy(result)
    #print(f"score={score}")
    #con_matrix = classifer.confusion_matrix(result)
    #print(f"confusion_matrix=${con_matrix}")
 
    

    print('| Experiement | Accuracy | Confusion Matrix | Comment |')
    print('|-------------|----------|------------------|---------|')
    print('| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |')

    #solution 1-3
    for i in range(3):
        classifier = DiabetesClassifier()
        result = classifier.predict(cols[i])
        score = classifier.calculate_accuracy(result)
        con_matrix = classifier.confusion_matrix(result)
        print(f'| Solution %d| {score} | {con_matrix.tolist()} | {cols[i]} |'%(i+1))

    classifier_best = classifier

    
    #solution 4
    features = ['pregnant', 'bmi', 'glucose', 'bp']
    classifier = DiabetesClassifier()
    classifier.bucketize_a_feature('bp')
    result = classifier.predict(features)
    score = classifier.calculate_accuracy(result)
    con_matrix = classifier.confusion_matrix(result)
    print(f'| Solution 4| {score} | {con_matrix.tolist()} | {features} bucketized bp |')

    #solution 5
    features = ['pregnant', 'bmi', 'glucose', 'bp']
    classifier = DiabetesClassifier()
    classifier.bucketize_a_feature('bmi')
    result = classifier.predict(features)
    score = classifier.calculate_accuracy(result)
    con_matrix = classifier.confusion_matrix(result)
    print(f'| Solution 5| {score} | {con_matrix.tolist()} | {features} bucketized bmi |')


    print('\nSolution 3 has the highest prediction accuracy on test dataset. ')
    result = classifier_best.predict(cols[i])
    score = classifier_best.calculate_accuracy(result)
    con_matrix = classifier_best.confusion_matrix(result)
    print('TP = ', con_matrix[0, 0])
    print('TN = ', con_matrix[1, 1])
    print('FP = ', con_matrix[0, 1])
    print('FN = ', con_matrix[1, 0])
    print('Recall =', con_matrix[0, 0] / (con_matrix[0, 0] + con_matrix[1, 0]))
    print('Precision = ', con_matrix[0,0] / (con_matrix[0,0] + con_matrix[0, 1]))

    #Using chi2 to exam each feature's correlation with the target
    classifier = DiabetesClassifier()
    cols =  ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    #result = classifier.predict(cols)
    test = SelectKBest(score_func=chi2, k=4)
    scaler = StandardScaler()
    X = scaler.fit_transform(classifier.pima[cols])
    fit = test.fit(abs(X), classifier.pima.label )
    print(cols)
    width = 10
    print('KBest scores of each feature(score function is chi2) \n', fit.scores_)
