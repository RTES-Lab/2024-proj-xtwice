#!/bin/bash

python main.py --dates 1105 1217 0108 --view F --axis z --input_feature z --save_log
python main.py --dates 1105 1217 0108 --view F --axis z --input_feature z_rms --save_log
python main.py --dates 1105 1217 0108 --view F --axis z --input_feature z_fused_features --save_log 