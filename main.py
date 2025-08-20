# Facial Recognition — LFW, 60/20/20 split, two models, full metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset (people with many images to avoid tiny classes)
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.5, color=False)  # downloads on first run
X = lfw.data    # (n_samples, n_features)
y = lfw.target  # integer labels
class_names = lfw.target_names
n_classes = len(class_names)
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {n_classes}, Names: {class_names}")

# 2) 60/20/20 stratified split: train / val / test (fixed seed for reproducibility)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)  # 0.2 each

# 3) Scale features for linear models (fit on train only to avoid leakage)
scaler = StandardScaler(with_mean=False)  # sparse-like images
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# 4) Models
logreg = OneVsRestClassifier(
    LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
)
tree = DecisionTreeClassifier(
    max_depth=25, min_samples_split=4, random_state=42
)

# 5) Train on train+val (common when hyperparams are fixed); alternatively tune on val then refit
logreg.fit(np.vstack([X_train_s.toarray() if hasattr(X_train_s, "toarray") else X_train_s,
                      X_val_s.toarray() if hasattr(X_val_s, "toarray") else X_val_s]),
           np.hstack([y_train, y_val]))
tree.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

# 6) Evaluate on test set
def evaluate(model, X_eval, y_eval, name, scaled=False):
    X_in = X_eval if not scaled else scaler.transform(X_eval)
    y_pred = model.predict(X_in if not hasattr(X_in, "toarray") else X_in.toarray())
    print(f"\n=== {name}: classification report (macro & weighted F1 highlighted) ===")
    print(classification_report(y_eval, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.show()

evaluate(logreg, X_test, y_test, "Logistic Regression (OvR)", scaled=True)
evaluate(tree,   X_test, y_test, "Decision Tree", scaled=False)

# 7) Optional: ROC curves (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
# LR scores
lr_scores = logreg.decision_function(scaler.transform(X_test))
# For tree, use predict_proba
tr_scores = tree.predict_proba(X_test)

def plot_roc(scores, y_bin, name):
    plt.figure()
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} — One-vs-Rest ROC")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

plot_roc(lr_scores, y_test_bin, "Logistic Regression")
plot_roc(tr_scores, y_test_bin, "Decision Tree")
# pip install opencv-python face_recognition numpy Pillow
