#!/bin/bash

for rate in 0.01 
do
    for epoch in {50..150..10}
    do
        python improse-layer4_metrics $rate $epoch > layer4'_'$rate'_'$epoch.txt
    done
done
