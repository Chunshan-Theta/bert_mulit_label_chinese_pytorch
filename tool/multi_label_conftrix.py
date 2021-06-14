from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# y_pred = [0,0,2,3,0,1,0]
# y_true = [0,1,2,0,0,1,0]

def multi_label_confusion_matrix(y_pred,y_true):
  y2_pred = []
  y2_true = []
  for y_p, y_t in zip(y_pred,y_true):
    for idx,(p,t) in enumerate(zip(y_p, y_t)):
      y2_true.append(idx)
      if p==1 and t==0:
        y2_pred.append(-1)
      elif p==0 and t==1:
        y2_pred.append(-2)
      else:
        y2_pred.append(idx)


  cf_matrix = confusion_matrix(y2_true, y2_pred)
  return cf_matrix

y_pred = [[1,0,1,0],[0,0,0,1]]
y_true = [[1,0,0,1],[0,1,0,1]]
cf_matrix = multi_label_confusion_matrix(y_pred,y_true)

labes_group_num = len(y_pred[0])
classes = ["type-2", "type-1"] + [f"Gropu-{idx}" for idx in range(0, labes_group_num)]
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')