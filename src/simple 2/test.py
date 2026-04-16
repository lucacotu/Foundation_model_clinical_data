import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

from data_loader import load_data
from preprocessing import clean_and_impute, prepare_cox_data_cv

def small_data():
    """60 train + 20 test rows, 4 features, binary event labels."""
    rng = np.random.default_rng(7)
    n = 80
    X = rng.standard_normal((n, 4)).astype(np.float32)
    durations = rng.uniform(1, 100, n).astype(np.float32)
    events = rng.integers(0, 2, n).astype(np.float32)
    return (
        X[:60], durations[:60], events[:60],   # train
        X[60:], durations[60:], events[60:],   # test
    )

def test_tabpfn(train_set,eval_set,head_type='cox'):
    from simple.models.tabpfn.model import TabPFNSurvPH

    X, y, t = train_set
    X_eval, y_eval, t_eval = eval_set

    ph = TabPFNSurvPH(
        head_type=head_type,
        num_durations=5,
        device="cuda",
    )
    ph.fit(X, t, y, epochs=2, batch_size=20, verbose=True)
    surv_df_eval = ph.predict_survival_df(X_eval)
    surv_df_train = ph.predict_survival_df(X)

    assert isinstance(surv_df_eval, pd.DataFrame), "Expected DataFrame output"
    _, n_subjects = surv_df_eval.shape
    assert n_subjects == len(X_eval), (
        f"Expected {len(X_eval)} subjects, got {n_subjects}"
    )

    ev_test = EvalSurv(surv_df_eval, t_eval, y_eval, censor_surv='km')
    ev_train = EvalSurv(surv_df_train, t, y, censor_surv='km')

    print("C-index Test:", ev_test.concordance_td())
    print("C-index Train:", ev_train.concordance_td())
    return ((ev_train.concordance_td(),ev_test.concordance_td()))

def test_tabicl(train_set,eval_set,head_type):
    from simple.models.tabicl.model import TabICLSurvPH

    X, y, t = train_set
    X_eval, y_eval, t_eval = eval_set

    # Preprocess with l'imputer di TabICL
    encoder = TransformToNumerical()
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    

    ph = TabICLSurvPH(
        head_type=head_type,
        num_durations=5,
        device="cuda",
        epochs=2,
    )
    ph.fit(X, t, y, epochs=2, batch_size=20, verbose=True)
    surv_df_eval = ph.predict_survival_df(X_eval)
    surv_df_train = ph.predict_survival_df(X)

    assert isinstance(surv_df_eval, pd.DataFrame), "Expected DataFrame output"
    n_times, n_subjects = surv_df_eval.shape
    assert n_subjects == len(X_eval), (
        f"Expected {len(X_eval)} subjects, got {n_subjects}"
    )
    assert n_times > 0, "Survival DataFrame has no time points"

    ev_test = EvalSurv(surv_df_eval, t_eval, y_eval, censor_surv='km')
    ev_train = EvalSurv(surv_df_train, t, y, censor_surv='km')

    print("C-index Test:", ev_test.concordance_td())
    print("C-index Train:", ev_train.concordance_td())
    return ((ev_train.concordance_td(),ev_test.concordance_td()))

TABDPT_CHECKPOINT="/home/lcotugno/Foundation_Model_clinical_data/src/models_diff/tabdpt1_1.pth"

def test_tabdpt(train_set,eval_set,head_type):
    from simple.models.tabdpt.model import TabDPTSurvPH

    X, y, t = train_set
    X_eval, y_eval, t_eval = eval_set

    ph = TabDPTSurvPH(
        head_type=head_type,
        num_durations=5,
        device="cuda",
        checkpoint_path=TABDPT_CHECKPOINT,
    )
    ph.fit(X, t, y, epochs=2, batch_size=20, verbose=True)
    surv_df_eval = ph.predict_survival_df(X_eval)
    surv_df_train = ph.predict_survival_df(X)

    assert isinstance(surv_df_eval, pd.DataFrame)
    _, n_subjects = surv_df_eval.shape
    assert n_subjects == len(X_eval)

    ev_test = EvalSurv(surv_df_eval, t_eval, y_eval, censor_surv='km')
    ev_train = EvalSurv(surv_df_train, t, y, censor_surv='km')

    print("C-index Test:", ev_test.concordance_td())
    print("C-index Train:", ev_train.concordance_td())
    return ((ev_train.concordance_td(),ev_test.concordance_td()))

if __name__ == "__main__":

    # 1. Load and clean the data
    data = load_data("OrmoniTirodei", "Dataset Sirbu")
    df = clean_and_impute("OrmoniTirodei", data)

    # 2. Extract specific features and targets (e.g. Mortality data)
    # And split into Train, Eval and Test sets
    df_mortality_train, df_mortality_eval = prepare_cox_data_cv(df)

    df_mortality_train = df_mortality_train.reset_index(drop=True)
    df_mortality_eval = df_mortality_eval.reset_index(drop=True)

    X_eval = df_mortality_eval.drop(columns=["Follow Up Data","Total mortality"])
    y_eval = df_mortality_eval["Total mortality"].values.astype(np.float32)
    t_eval = df_mortality_eval["Follow Up Data"].values.astype(np.float32)

    X = df_mortality_train.drop(columns=["Follow Up Data","Total mortality"])
    y = df_mortality_train["Total mortality"].values.astype(np.float32)
    t = df_mortality_train["Follow Up Data"].values.astype(np.float32)

    result_train_tabpfn, result_test_tabpfn = test_tabpfn((X, y, t),(X_eval, y_eval, t_eval),'deephit')
    result_train_tabdpt, result_test_tabdpt = test_tabdpt((X, y, t),(X_eval, y_eval, t_eval),'deephit')
    result_train_tabicl, result_test_tabicl = test_tabicl((X, y, t),(X_eval, y_eval, t_eval),'deephit')

    print(result_train_tabpfn)
    print(result_test_tabpfn)
    print(result_train_tabdpt)
    print(result_test_tabdpt)
    print(result_train_tabicl)
    print(result_test_tabicl)