# -*- coding: utf-8 -*-


f1 = open('roc_result_layer2','w')


f2 = open('train_data.csv','r')

while True:
    line = f2.readline()
    #print "11L", line
    if line:
        # line = line.replace('\n','')
        if len(line)>0:
            f1.write(line[0]+'\r\n')
           # print '16L', line[0]
    else:
        break
    
f2.close()
f1.close()
       
    
    