#!/bin/bash

for rate in 0.01 0.05 0.001 0.005 0.0001 0.0005 0.00001 0.00005
do
    for epoch in {50..150..10}
    do
        python improse-layer2_metrics.py $rate $epoch > layer2'_'$rate'_'$epoch.txt
    done
done
