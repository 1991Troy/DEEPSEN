#!/bin/bash

for rate in 0.05 
do
    for epoch in {50..150..10}
    do
        python improse-layer3_metrics.py $rate $epoch > layer3'_'$rate'_'$epoch.txt
    done
done
