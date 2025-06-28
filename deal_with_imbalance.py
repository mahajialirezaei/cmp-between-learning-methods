from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pre_processing_data import pre_processing
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score

def smote_tech(X_train_processed, y_train):
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    return X_train_smote, y_train_smote, class_weights, weights


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
    return models





def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'Precision': precision_score(y_test, y_pred, pos_label=1),
        'Recall': recall_score(y_test, y_pred, pos_label=1),
        'F1-Score': f1_score(y_test, y_pred, pos_label=1),
        'AUC-PR': average_precision_score(y_test, y_proba) if y_proba is not None else None,
        'G-Mean': geometric_mean_score(y_test, y_pred, pos_label=1)
    }
    return metrics




def geometric_mean_score(y_true, y_pred, pos_label=1):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)



if __name__ == '__main__':
    X_train_processed, X_test_processed, y_train, y_test = pre_processing()
    X_train_smote, y_train_smote, class_weights, weights = smote_tech(X_train_processed, y_train)
    models = define_models(X_train_processed, y_train, X_train_smote, y_train_smote, class_weights, weights)
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test_processed, y_test)

