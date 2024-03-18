import pickle
from svm import SVM
from random_forest import DecisionTreeClassifier, DecisionNode, RandomForestClassifier

models = pickle.load(open("./models.pickle", "rb"))
label_encoders = models['label_encoders']


for k in label_encoders:
    le = label_encoders[k]
    print(f"for key {k}\nlabel encoder classes: {le.classes_}\n\n")
    # print(f"for key {k}, label encoder classes: {le.transform(['f', 'm'])}")
