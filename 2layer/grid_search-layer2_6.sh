#!/bin/bash

for rate in  0.0005 
do
    for epoch in {50..150..10}
    do
        python improse-layer2_metrics.py $rate $epoch > layer2'_'$rate'_'$epoch.txt
    done
done
