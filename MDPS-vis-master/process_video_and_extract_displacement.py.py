import os

i:int = 7
bearing_type: str = "IR"
os.system(f"python ./example/get_roi.py -fname ./input/0530_30204_{bearing_type}_1200RPM_120fps_{i}.mov -f 120 -o ./test/0530_30204_{bearing_type}_1200RPM_120fps_{i}.json")
os.system(f"python ./example/extract_displacement.py -fname ./input/0530_30204_{bearing_type}_1200RPM_120fps_{i}.mov -f 120 -skip 0 -o ./output/0530_30204_{bearing_type}_1200RPM_120fps_{i} -fo 14 -flb 0.01 -fub 1.0 -a 0 -roi ./test/0530_30204_{bearing_type}_1200RPM_120fps_{i}.json ")

'''
B, 1 :     tracker = MarkerCentroidTracker((90, 75, 90), (128, 255, 255))
B, 2 :    tracker = MarkerCentroidTracker((103, 189, 0), (179, 255, 255))
B, 3 :     tracker = MarkerCentroidTracker((100, 167, 219), (179, 255, 255))
B, 4 :     tracker = MarkerCentroidTracker((100, 145, 140), (179, 255, 255))
B, 5 :     tracker = MarkerCentroidTracker((25, 125, 125), (100, 255, 255))
B, 6 :     tracker = MarkerCentroidTracker((100, 200, 120), (179, 255, 255))
B, 7 :     tracker = MarkerCentroidTracker((100, 145, 140), (179, 255, 255))

IR, 1 :     tracker = MarkerCentroidTracker((28, 75, 75), (121, 255, 255))
IR, 2 :     tracker = MarkerCentroidTracker((100,140, 100), (179, 255, 255))
IR, 3 :     tracker = MarkerCentroidTracker((100, 130, 170), (179, 255, 255))
IR, 4 :     tracker = MarkerCentroidTracker((100, 145, 140), (179, 255, 255))
IR, 5 :     tracker = MarkerCentroidTracker((25, 125, 125), (100, 255, 255))
IR, 6 :     tracker = MarkerCentroidTracker((100, 200, 120), (179, 255, 255))
'''