#!/bin/bash

batches="128 512 1024”
training_time=“1 5 10”
for batch in $batches; do
    for time in $training_time; do
		nvprof python main.py cifar10 $time $batch 1
		nvprof python main.py cifar10 $time $batch 0
    done
done