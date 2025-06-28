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



def define_models(X_train_processed, y_train, X_train_smote, y_train_smote, class_weights, weights):
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            scale_pos_weight=weights[1] / weights[0],  # برای کلاس اقلیت
            random_state=42,
            eval_metric='aucpr'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            objective='binary'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0
        )
    }

    for name, model in models.items():
        model.fit(X_train_processed, y_train)

        if name != 'CatBoost':
            model.set_params(**{'class_weight': class_weights})
        model.fit(X_train_processed, y_train)

        model.fit(X_train_smote, y_train_smote)

        model.set_params(**{'class_weight': None})
        model.fit(X_train_smote, y_train_smote)


