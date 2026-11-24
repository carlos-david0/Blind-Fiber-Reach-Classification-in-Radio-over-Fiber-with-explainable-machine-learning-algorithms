import optuna
import pandas as pd
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, label_binarize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures

def calibration_metrics(y_true, y_prob, class_labels, n_bins=10):
    """ECE, MCE, and per-bin data for a multiclass classifier."""
    y_true = np.asarray(y_true)
    conf = y_prob.max(axis=1)
    pred = np.asarray(class_labels)[y_prob.argmax(axis=1)]
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    bin_conf, bin_acc, bin_count = [], [], []

    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if mask.sum() == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        gap = abs(acc_b - conf_b)
        ece += gap * mask.mean()
        mce = max(mce, gap)
        bin_conf.append(conf_b)
        bin_acc.append(acc_b)
        bin_count.append(mask.sum())

    return ece, mce, np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


def plot_reliability(bin_conf, bin_acc, ece, path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(bin_conf, bin_acc, "o-", color="darkorange", label=f"SVM (ECE = {ece:.3f})")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.savefig(path)
    plt.close()

n_classes = 4
classes = [0,1,2,3]
name_classes = [0, 2, 20, 40]

# Define objective function to minimize/maximize
def objective(trial):
    # Suggest hyperparameter values (Log scale for C/gamma)
    c = trial.suggest_float("C", 1e-3, 1e3, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    #kernel = trial.suggest_categorical("kernel", ["linear"])
    # Train and evaluate
    clf = SVC(C=c, gamma=gamma, kernel=kernel)
    return sklearn.model_selection.cross_val_score(
        clf, train_data, train_target, cv=5).mean()

# Run optimization
data = pd.read_csv("../datos tsfel/USPL/caracteristicas_raw50000.csv")
targets = data["distance"].to_list()
data = data.drop(columns=["distance", "OSNR"])

sizes = [0.8]
accuracies = []
for x in sizes:
    # 1) split FIRST, on raw features
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, targets, train_size=x, shuffle=True, random_state=42)

    # 2) feature selection, fit on TRAIN only
    const = DropConstantFeatures(tol=1.0)
    corr = DropCorrelatedFeatures(method="pearson", threshold=0.7)
    train_data = corr.fit_transform(const.fit_transform(train_data))
    test_data = corr.transform(const.transform(test_data))   # same columns, no refit
    selected_cols = list(train_data.columns)
    print(f"Retained features P = {len(selected_cols)}")      # paper says 73

    # 3) scaling, fit on TRAIN only
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    print(f"Best: {study.best_params} (Acc: {study.best_value:.4f})")

    clf = SVC(C=study.best_params['C'], gamma=study.best_params['gamma'], kernel=study.best_params['kernel'], probability=True)
    clf.fit(train_data, train_target)
    predicts = clf.predict(test_data)
    accuracy = clf.score(test_data, test_target)
    y_score = clf.predict_proba(test_data)
    y_test_bin = label_binarize(test_target, classes=classes)

        # ---- Calibration plot (one-vs-rest) ----
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 20})
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green"])
    for i, color in zip(range(n_classes), colors):
        frac_pos, mean_pred = calibration_curve(
            y_test_bin[:, i], y_score[:, i], n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, "s-", color=color, lw=2,
                 label=f"{name_classes[i]}km")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper left", fontsize=14)
    plt.title("SVG", fontsize=40, loc="left")
    plt.tight_layout()
    plt.savefig(f"../resultados/uspl/60000/svm/calibration_plot_{x}.svg")
    plt.close()

        # ---- Calibration ----
    ece, mce, bin_conf, bin_acc, bin_count = calibration_metrics(
        test_target, y_score, clf.classes_, n_bins=10)
    brier = np.mean(np.sum((y_score - y_test_bin) ** 2, axis=1))
    nll = log_loss(test_target, y_score, labels=classes)

    print(f"ECE: {ece:.4f} | MCE: {mce:.4f} | Brier: {brier:.4f} | NLL: {nll:.4f}")

    plot_reliability(bin_conf, bin_acc, ece,
                     f"../resultados/uspl/60000/svm/reliability_{x}.svg")

    pd.DataFrame([{"train_size": x, "ECE": ece, "MCE": mce,
                   "Brier": brier, "NLL": nll}]).to_csv(
        f"../resultados/uspl/60000/svm/calibration_{x}.csv", index=False)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print(f"train size: {x * 100:.2f}%")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    fold_result = sklearn.model_selection.cross_val_score(clf, train_data, train_target, cv=5)
    f1_result = sklearn.model_selection.cross_val_score(clf, train_data, train_target, cv=5, scoring="f1_macro")
    print(f"k folds: {fold_result}")
    weights = permutation_importance(clf, train_data, train_target, n_repeats=10, random_state=42)
    feature_list = []
    importance_list = []
    for i in weights.importances_mean.argsort()[::-1]:
        feature_list.append(selected_cols[i])
        importance_list.append(weights.importances_mean[i])
    df = pd.DataFrame({'feature': feature_list, 'importance': importance_list})
    df = df.sort_values(by='importance', ascending=False)
    df.to_csv(f"../resultados/uspl/60000/svm/feature_importance_{x}.csv", index=False)

    accuracies.append(accuracy)
    test_data = pd.DataFrame(test_data)
    test_data["predictions"] = predicts
    test_data["transmited_symbol"] = test_target
    test_data.to_csv(f"../resultados/uspl/60000/svm/results_{x}.csv", index=False)
    df = pd.DataFrame([study.best_params])
    df.to_csv(f"../resultados/uspl/60000/SVM/parameters_{x}.csv", index=False)
    df = pd.DataFrame(columns=["0f", "1f", "2f", "3f", "4f", "sizes", "type"])
    df.loc[len(df)] = np.append(fold_result, [i, "Accuracy"])
    df.loc[len(df)] = np.append(f1_result, [i, "f1"])
    df.to_csv(f"../resultados/uspl/60000/svm/k_folds_{x}.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 20})
    plt.plot(fpr["micro"], tpr["micro"],
            label = f"M.Av (AUC = {roc_auc["macro"]:.2f})",
            color="navy", linestyle=":", linewidth=3)
        
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label= f"{name_classes[i]}km (AUC {roc_auc[i]:.2f})")
    
    plt.plot([0,1], [0,1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.title("(b)", fontsize=40, loc="left")
    plt.legend(loc="lower right")
    plt.savefig("../resultados/uspl/60000/svm/roc.svg")

df = pd.DataFrame({'sizes': sizes, 'accuracy': accuracies})
df.to_csv(f"../resultados/uspl/60000/svm/acuracy.csv", index=False)
