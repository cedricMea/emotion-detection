
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd

# sklearn.metrics.log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)

def train_fn(X_data, y_data):


    parameters = {
        'penalty':['l2','l1'], 
        'C':[10, 100, 1000],
        'solver': ["liblinear", "saga"]
        #'solver': ['saga','lbfgs','newton-cg', 'liblinear']
        }


    model =  LogisticRegression(multi_class='ovr')

    all_labels=np.unique(y_data)

    # def multinomial_log_loss(y_true, y_pred):
    #     y_pred = pd.get_dummies(y_pred)
    #     return log_loss(y_true, y_pred, labels=all_labels)

    best_clf = GridSearchCV(
        estimator=model, 
        param_grid=parameters,
        scoring={
            "log_loss": make_scorer(log_loss, greater_is_better=False, needs_proba=True, labels=all_labels),
            "accuracy": "accuracy"
        },
        refit="log_loss",
        cv=5,
        verbose=10,
        n_jobs=2
        )


    best_clf = best_clf.fit(
        X=X_data,
        y=y_data
    )

    print(best_clf.best_params_)

    return best_clf




def eval_fn(sentence, model_dict):

    lr_model = model_dict.get("model_pipeline")
    treshold = model_dict.get("lr_treshold")
    sentence_df = pd.DataFrame({"text":[sentence]})
    preds = lr_model.predict_proba(sentence_df)
    # print(preds, treshold)

    return(preds, treshold)





