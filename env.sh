#!/bin/sh

eval "$(/idiap/temp/ibmahmoud/miniconda3/bin/conda shell.bash hook)"

conda activate meerkats

export PYTHONPATH=/idiap/project/evolang/meerkats_imen/evolang_meerkats_calls_classification:$PYTHONPATH
