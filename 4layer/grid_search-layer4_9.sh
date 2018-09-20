#!/bin/bash

for rate in 0.00005
do
    for epoch in {160..200..10}
    do
        python improse-layer4_metrics.py $rate $epoch > layer4'_'$rate'_'$epoch.txt
    done
done
