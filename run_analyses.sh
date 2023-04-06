#!/bin/bash

# Run analyses
python3 /home/scripts/prep_dataframes.py

python3 /home/scripts/covar_correction.py

python3 /home/scripts/HC_normalization.py

python3 /home/scripts/graph_analysis.py

python3 /home/scripts/group_stats.py

python3 /home/scripts/correlations.py

python3 /home/scripts/linear_regressions.py



#tar -czvf /home/output/output.tar.gz /home/
