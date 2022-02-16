#!/bin/bash

python benchmark_ml.py --benchmark_name 'BBBscore'

python benchmark_ml.py --benchmark_name 'RF'

python benchmark_ml.py --benchmark_name 'MLP'

python benchmark_ml.py --benchmark_name 'RF_PCP'

python benchmark_ml.py --benchmark_name 'MLP_PCP'

python benchmark_ml.py --benchmark_name 'AttentiveFP'
