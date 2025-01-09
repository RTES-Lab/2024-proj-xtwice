#!/bin/bash

python draw_hist.py --dates 1105 1217 0108 --view F --axis z --mode original
python draw_hist.py --dates 1105 --view F --axis z --mode original
python draw_hist.py --dates 1217 --view F --axis z --mode original
python draw_hist.py --dates 0108 --view F --axis z --mode original

python draw_hist.py --dates 1105 1217 0108 --view F --axis z --mode train
