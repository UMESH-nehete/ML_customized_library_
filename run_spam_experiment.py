import numpy as np

from my_ml_lib.datasets import load_spambase
from my_ml_lib.model_selection._split import train_test_split
from my_ml_lib.model_selection._kfold import KFold
from my_ml_lib.preprocessing._data import StandardScaler
from my_ml_lib.linear_models.classification._logistic import LogisticRegression

# 1) load data
X, y = load_spambase()

# 2) split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# alpha candidates
alphas = [0.01, 0.1, 1, 10, 100]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cross_val_score(alpha, Xtrain, ytrain):
    scores = []
    for train_idx, val_idx in kf.split(Xtrain):
        Xtr, Xval = Xtrain[train_idx], Xtrain[val_idx]
        ytr, yval = ytrain[train_idx], ytrain[val_idx]

        clf = LogisticRegression(alpha=alpha, max_iter=500)
        clf.fit(Xtr, ytr)
        scores.append(clf.score(Xval, yval))
    return np.mean(scores)

# 3a) raw
best_raw_score = -1
best_raw_alpha = None
for a in alphas:
    sc = cross_val_score(a, X_train, y_train)
    if sc > best_raw_score:
        best_raw_score = sc
        best_raw_alpha = a

clf_raw = LogisticRegression(alpha=best_raw_alpha, max_iter=500)
clf_raw.fit(X_train, y_train)

train_err_raw = 1 - clf_raw.score(X_train, y_train)
test_err_raw  = 1 - clf_raw.score(X_test, y_test)

# 3b) standardized
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

best_std_score = -1
best_std_alpha = None
for a in alphas:
    sc = cross_val_score(a, X_train_std, y_train)
    if sc > best_std_score:
        best_std_score = sc
        best_std_alpha = a

clf_std = LogisticRegression(alpha=best_std_alpha, max_iter=500)
clf_std.fit(X_train_std, y_train)

train_err_std = 1 - clf_std.score(X_train_std, y_train)
test_err_std  = 1 - clf_std.score(X_test_std, y_test)

print("RESULT TABLE")
print("##########################################################")
print(f"RAW        best alpha={best_raw_alpha}, train_error={train_err_raw}, test_error={test_err_raw}")
print(f"STANDARD   best alpha={best_std_alpha}, train_error={train_err_std}, test_error={test_err_std}")
