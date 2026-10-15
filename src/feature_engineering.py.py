from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'sales.csv'

if not DATA_PATH.exists():
    raise SystemExit('Dataset missing: run python src/train.py first')

df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
required = ['price', 'ad_budget', 'discount_pct', 'weekend', 'units_sold']
if not set(required).issubset(df.columns):
    raise ValueError(f'Missing columns: {set(required) - set(df.columns)}')

df['effective_price'] = df['price'] * (1 - df['discount_pct'])

end_train = int(0.60 * len(df))
end_val = int(0.80 * len(df))
train, val, test = df.iloc[:end_train], df.iloc[end_train:end_val], df.iloc[end_val:]

base_features = ['price', 'ad_budget', 'discount_pct', 'weekend']
engineered_features = base_features + ['effective_price']

def build_pipeline(features):
    numeric = [feature for feature in features if feature != 'weekend']
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore')),
    ])
    processor = ColumnTransformer([
        ('numeric', numeric_pipe, numeric),
        ('categorical', categorical_pipe, ['weekend']),
    ])
    return Pipeline([
        ('preprocessing', processor),
        ('regression', LinearRegression()),
    ])

def evaluate(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

choices = {
    'base': base_features,
    'engineered': engineered_features,
}
val_scores = {}
for name, features in choices.items():
    model = build_pipeline(features)
    model.fit(train[features], train['units_sold'])
    val_scores[name] = evaluate(val['units_sold'], model.predict(val[features]))

baseline_val = evaluate(
    val['units_sold'],
    np.full(len(val), train['units_sold'].mean()),
)

chosen = min(val_scores, key=lambda name: val_scores[name]['MAE'])
chosen_features = choices[chosen]
trainval = df.iloc[:end_val]
final_model = build_pipeline(chosen_features)
final_model.fit(trainval[chosen_features], trainval['units_sold'])

test_scores = evaluate(test['units_sold'], final_model.predict(test[chosen_features]))
baseline_test = evaluate(test['units_sold'], np.full(len(test), trainval['units_sold'].mean()))

report = {
    'experiment': '002',
    'dataset': 'synthetic retail sales',
    'split': {'train': len(train), 'validation': len(val), 'test': len(test)},
    'validation_scores': val_scores,
    'baseline_validation': baseline_val,
    'selected_model': chosen,
    'selected_features': chosen_features,
    'test_scores': test_scores,
    'baseline_test': baseline_test,
}
(ROOT / 'reports').mkdir(exist_ok=True)
(ROOT / 'models').mkdir(exist_ok=True)
(ROOT / 'reports' / 'experiment_002_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
joblib.dump(final_model, ROOT / 'models' / 'sales_pipeline_v02.joblib')
print(json.dumps(report, indent=2))
