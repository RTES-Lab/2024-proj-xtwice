import os

i:int = 2
bearing_type: str = "B"
os.system(f"python ./example/get_roi.py -fname ./input/0530_30204_{bearing_type}_1200RPM_120fps_{i}.mov -f 120 -o ./test/0530_30204_{bearing_type}_1200RPM_120fps_{i}.json")
os.system(f"python ./example/extract_displacement.py -fname ./input/0530_30204_{bearing_type}_1200RPM_120fps_{i}.mov -f 120 -skip 0 -o ./output/0530_30204_{bearing_type}_1200RPM_120fps_{i} -fo 14 -flb 0.01 -fub 1.0 -a 0 -roi ./test/0530_30204_{bearing_type}_1200RPM_120fps_{i}.json ")