from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import tree

dict_classifiers = {
    # "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(penalty='l1', solver='liblinear', C=1.0),
    # "Logistic Regression": LogisticRegression(),
    # "Nearest Neighbors": KNeighborsClassifier(),
    # "Decision Tree": DecisionTreeClassifier(),
    # "Linear SVM": SVC(),
    # "Neural Net": MLPClassifier(alpha = 1),
    # "AdaBoost": AdaBoostClassifier(),
    # "Gaussian Process": GaussianProcessClassifier()
}
