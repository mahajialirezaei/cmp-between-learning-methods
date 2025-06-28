import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


def pre_processing():
    data = pd.read_csv('dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = encoder.fit_transform(X_train.select_dtypes(include=['object']))
    X_test_cat = encoder.transform(X_test.select_dtypes(include=['object']))

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train.select_dtypes(exclude=['object']))
    X_test_num = scaler.transform(X_test.select_dtypes(exclude=['object']))

    X_train_processed = np.concatenate([X_train_num, X_train_cat], axis=1)
    X_test_processed = np.concatenate([X_test_num, X_test_cat], axis=1)
    return X_train_processed, X_test_processed, y_train, y_test, num_imputer, cat_imputer