import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from csp import *
from data import *
from track import *
from pbm import *
from utils import *

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-fname", "--filename", dest="filename", required=True, type=str)
    parser.add_argument("-f", "--fps", dest="fps", required=True, type=int)
    parser.add_argument("-o", "--output", dest="output", required=True, type=str)
    args = parser.parse_args()

    stream = VideoFileReader(args.filename, 24, False, args.fps)
    img = stream.read()

    idx_crop, idx_pbm, coord = region_from_ui(img)
    region_to_json(idx_crop, idx_pbm, coord, args.output)


