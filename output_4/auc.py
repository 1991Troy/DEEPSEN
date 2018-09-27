#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt
label=[]
result=[]

f1=open('layer4_0.00005_110.output','r')
count = 0
while True:
    line=f1.readline().replace('\n','')
    if line and len(line)>3:
        label.append(float(line.split(' ')[0]))
        result.append(float(line.split(' ')[1]))
        count += 1
    else:
        break
print(count)
false_positive_rate,true_positive_rate,thresholds=roc_curve(label,result)
roc_auc=auc(false_positive_rate, true_positive_rate)


plt.title('Receiver Operating Characteristic')
l1, = plt.plot(false_positive_rate, true_positive_rate,'b')


plt.plot([0,1],[0,1], color = 'r', linestyle = '--')

plt.xlim([0.00,1.00])
plt.ylim([0.00,1.00])
plt.legend(handles = [l1], labels = ['AUC = 0.9820'], loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

f1.close()