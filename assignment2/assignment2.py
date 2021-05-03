from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import GridSearchCV, train_test_split

def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60, data_home = '~/git/qw255/cmpe255-spring21/assignment2/')
    print('data loaded')
    print(dir(faces))
    print(faces.data.shape)       
    print(faces.images.shape)       
    print(faces.target.shape)       
    print(faces.target_names)
    return faces


def grid_search(X_train, y_train, X_test, y_test):
    pipeline = Pipeline(steps=[
     ('pca', pca),
     ('svc', svc)
    ])

    param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma':[1, 0.1, 0.01, 0.001, 0.0001], 
    'svc__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(pipeline, param_grid)

    # fitting the model for grid search 
    grid.fit(X_train, y_train) 

    # print best parameter after tuning 
    print(grid.best_params_) 
    grid_predictions = grid.predict(X_test) 

    # print classification report 
    print(classification_report(y_test, grid_predictions)) 
    return grid_predictions

def plot_images(X_test, y_test, predictions, target_names):
    arr = np.arange(258)
    np.random.shuffle(arr)
    n = 0
    fig, axs = plt.subplots(4, 6, figsize = (12, 12))
    for i in range(4):
      for j in range(6):
          index = arr[n]
          pred = predictions[index]
          img = X_test[index]
          true = y_test[index]
          if pred == true:
            axs[i, j].imshow(img.reshape(62, 47), cmap = 'gray')
          if pred != true:
            axs[i,j].imshow(img.reshape(62, 47), cmap = 'inferno')
          axs[i, j].set_title(target_names[true])
          n = n + 1
    plt.savefig('plot.png')
    plt.show()
    pass


pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
faces = load_data()

#1. Split the data into a training and testing set.
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size = 0.2)
print(X_train.shape)
print(X_test.shape)

#2. Use a [grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 
#cross-validation to explore combinations of [parameters](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) 
#to determine the best model: 
predictions = grid_search(X_train, y_train, X_test, y_test)



#3. Draw a 4x6 subplots of images using names as label with color black for correct instances and red for incorrect instances.
plot_images(X_test, y_test, predictions, faces.target_names)
print('4*6 subplots of images saved to file: ', 'plots.png')
