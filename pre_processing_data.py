import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def pre_processing():
    data = pd.read_csv('dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    # تقسیم داده
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    # تعریف ستون‌های عددی و طبقه‌ای
    numeric_features = ['transaction_amount', 'transaction_time', 'customer_history',
                        'feature4', 'feature5', 'feature6']
    categorical_features = ['merchant_category', 'location_mismatch', 'device_used']

    # ایجاد pipeline پیش‌پردازش
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # اعمال پیش‌پردازش
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed, y_train, y_test