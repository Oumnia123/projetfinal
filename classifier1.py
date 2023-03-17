from pandas import read_csv
from sklearn.model_selection import train_test_split # Splitting technique
from sklearn.linear_model import LogisticRegression #Algorithm, classifier, model
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.svm import SVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNearestNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Import dataset
dataset = load_iris()

model1=LogisticRegression()
model2=DecisionTreeClassifier()
model3=RandomForestClassifier()
model4= KNearestNeighborsClassifier()
model5=SVM()
# Save model and transform
import pickle
# save model
model1_file = "Model_1.pickle"

pickle.dump(model1,open(model1_file, 'wb'))

model2_file = "Model_2.pickle"

pickle.dump(model2,open(model2_file, 'wb'))

model3_file = "Model_3.pickle"

pickle.dump(model3,open(model3_file, 'wb'))

model4_file = "Model_4pickle"

pickle.dump(model4,open(model4_file, 'wb'))

model5_file = "Model_5.pickle"

pickle.dump(model5,open(model5_file, 'wb'))
