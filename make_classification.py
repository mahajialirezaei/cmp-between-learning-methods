import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=10000, n_features=6, n_informative=4,
                          n_redundant=1, n_classes=2, weights=[0.98, 0.02],
                          random_state=42)

df = pd.DataFrame(X, columns=['transaction_amount', 'transaction_time',
                             'customer_history', 'feature4', 'feature5', 'feature6'])

df['merchant_category'] = np.random.choice(['retail', 'food', 'travel', 'entertainment'], size=10000)
df['location_mismatch'] = np.random.choice(['yes', 'no'], size=10000, p=[0.1, 0.9])
df['device_used'] = np.random.choice(['mobile', 'desktop', 'tablet'], size=10000)

df['target'] = y

df.to_csv('dataset.csv', index=False)