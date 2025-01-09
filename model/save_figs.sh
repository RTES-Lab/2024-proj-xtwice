#!/bin/bash

python draw_hist.py --dates 1105 1217 0108 --view F --axis z --input_feature z_fused_features --save_figs
python draw_hist.py --dates 1105 --view F --axis z --input_feature z_fused_features --save_figs
python draw_hist.py --dates 1217 --view F --axis z --input_feature z_fused_features --save_figs
python draw_hist.py --dates 0108 --view F --axis z --input_feature z_fused_features --save_figs
