from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import  numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import psutil
import pandas as pd
from sklearn.metrics import  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
)

class_names = ['Abstract_Expressionism','Baroque','Cubism', 'Fauvism', 'Impressionism','Minimalism','Naive_Art_Primitivism','Pointillism','Rococo','Ukiyo_e']
num_classes = 10

def kfold_crossvalidation(x_train, y_train, cv, model, scaller = 0):
  cv_scores = []
  acc = []
  rec = []
  pres = []
  f1a = []

  train_times = []
  memory_usages = []
  predict_times = []
  all_conf_matrices = []
  all_reports = []
  if scaller == 1:
      scaler = StandardScaler()
      x_train = scaler.fit_transform(x_train)

  skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
      start_time = time.time()

      X_train_fold, X_val_fold = x_train[train_idx], x_train[val_idx]
      y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
      model.fit(X_train_fold, y_train_fold)
      end_time = time.time()
      train_times.append(end_time - start_time)


      start_time = time.time()
      y_pred = model.predict(X_val_fold)
      end_time = time.time()
      predict_times.append(end_time - start_time)


      f1 = f1_score(y_val_fold, y_pred,  average='weighted')
      precision = precision_score(y_val_fold, y_pred, average='weighted')
      recall = recall_score(y_val_fold, y_pred, average='weighted')
      accuracy = accuracy_score(y_val_fold, y_pred)
      cv_scores.append({"f1" : f1, "precision": precision ,"recall":recall,"accuracy" : accuracy})
      f1a.append(f1)
      acc.append(accuracy)
      rec.append(recall)
      pres.append(precision)
      # conf_matrix = confusion_matrix(y_val_fold, y_pred)
      # all_conf_matrices.append(conf_matrix)
      # report = classification_report(y_val_fold, y_pred, output_dict=True)
      # all_reports.append(report)

  # avg_report = {label: {metric: np.mean([rep[label][metric] for rep in all_reports])
  #                       for metric in all_reports[0][label]}
  #               for label in all_reports[0]}
  # avg_conf_matrix = np.mean(all_conf_matrices, axis=0)

  data = {
    "Fold": list(range(1, cv + 1)),
    "Precision": pres,
    "Recall": rec,
    "F1": f1a,
    "Accuracy": acc,
    "Train time": train_times,
    "Prediction time": predict_times,


    }
  #print(data)

  df = pd.DataFrame(data)

  avg_metrics = {
    "Fold": "Average",
    "Precision": np.mean(pres),
    "Recall": np.mean(rec),
    "F1": np.mean(f1a),
    "Accuracy": np.mean(acc),
    "Train time": np.mean(train_times),
    "Prediction time": np.mean(predict_times),
  }
  std_metrics = df.std(numeric_only=True)

  df = pd.concat([df, pd.DataFrame([avg_metrics]), pd.DataFrame([std_metrics])], ignore_index=True)
  #print(df)
  return df

def precision_recall_draw(y_true, y_score):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve

    y_test_bin = label_binarize(y_true, classes=[i for i in range(len(class_names))])

    plt.figure(figsize=(15, 10))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def roc_draw(y_true, y_score):
  from sklearn.preprocessing import label_binarize
  plt.figure(figsize=(15, 10))
  y_test_bin = label_binarize(y_true, classes=[i for i in range(len(class_names))])


  for i, class_name in enumerate(class_names):
     fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
     roc_auc = auc(fpr, tpr)
     plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')


  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic Curve')
  plt.legend(loc="best")
  plt.grid(True)
  plt.show()


def evaluate_model(y_true, y_pred, y_pred_proba):
    report_dict = classification_report(y_pred, y_true, target_names=class_names, output_dict=True)
    metrics_df2 = pd.DataFrame(report_dict).transpose()
    print(metrics_df2)
    print("-----------------------------------------------------------------------------------------------------------")

    precision_recall_draw(y_true, y_pred_proba)
    roc_draw(y_true, y_pred_proba)

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred), display_labels=class_names)
    disp.plot(cmap="cividis", xticks_rotation="vertical")
