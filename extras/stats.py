import os
from PIL import Image
s= ""
# define the class labels
class_labels = {
    "Pedestrian": 0,
    "Person on vehicle": 1,
    "Car": 2,
    "Bicycle": 3,
    "Motorbike": 4,
    "Non motorized vehicle": 5,
    "Static person": 6,
    "Distractor": 7,
    "Occluder": 8,
    "Occluder on the ground": 9,
    "Occluder full": 10,
    "Reflection": 11
}

# nums= [5*i for i in range(1, 1000)]

nums= [5*i for i in range(1, 100)]
numsVal= [50*i for i in range(1, 100)]

def stats(mot16_folder):
    for sequence_folder in os.listdir(mot16_folder):
	    # construct the path to the gt.txt file and img1 folder
       # construct the path to the gt.txt file and img1 folder
        gt_file = os.path.join(mot16_folder, sequence_folder, "gt", "gt.txt")
        img_dir = os.path.join(mot16_folder, sequence_folder, "img1")
        # out_dir= os.path.join(mot16_folder, sequence_folder, "out")
        # out_dir= os.path.join(mot16_folder, "outnewVal")
        s= sequence_folder+"_"
        if(sequence_folder==".DS_Store" or sequence_folder=="out1" or sequence_folder=="out" or sequence_folder=="out2" or sequence_folder=="outnew"):
            continue
    # convert the MOT16 format to YOLO format
    print(sequence_folder + " started")
    find_statistics(gt_file, img_dir)
    print(sequence_folder + " Completed")

def  find_statistics(gt_file, img_dir):
    num_frames= 0
    with open(gt_file, "r") as f:
        lines = f.readlines()
    freqs= {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}
    # loop over each line in the ground truth file
    num= 1
    for line in lines:
        line = line.strip().split(",")
        frame_num = int(line[0])
        num_frames= max(num_frames, frame_num)
        if(frame_num not in nums):
            continue
        track_id = int(line[1])
        x, y, w, h = float(line[2]), float(line[3]), float(line[4]), float(line[5])
        obj_class = line[7].strip()
        # label = class_labels[obj_class]
        label= int(obj_class)-1
        freqs[label]+=1
        num+=1
        
    class_counts= []
    for key in freqs:
        class_counts.append(freqs[key])
    return class_counts, num_frames
