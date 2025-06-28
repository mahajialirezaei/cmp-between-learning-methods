from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def smote_tech(X_train_processed, y_train):
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
