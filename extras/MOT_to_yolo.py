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
freqs= {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}
nums= [10*i for i in range(1, 100)]
numsVal= [50*i for i in range(1, 100)]

def convert_to_yolo_format(gt_file, img_dir, output_dir):
    # create the img1_labels and images folders if they don't exist
    label_dir = os.path.join(output_dir, "labels")
    img_dir_out = os.path.join(output_dir, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(img_dir_out, exist_ok=True)
    # open the ground truth file
    with open(gt_file, "r") as f:
        lines = f.readlines()

    # loop over each line in the ground truth file
    num= 1
    for line in lines:
        line = line.strip().split(",")
        frame_num = int(line[0])
        if(frame_num not in numsVal):
            continue
        track_id = int(line[1])
        x, y, w, h = float(line[2]), float(line[3]), float(line[4]), float(line[5])
        obj_class = line[7].strip()
        # label = class_labels[obj_class]
        label= int(obj_class)-1
        # print(label)
        # visibility = float(line[-1])
        # confidence = float(line[6])
        # if(visibility<1 or confidence<1):
        #     continue
        # create the YOLO format label file in the labels folder
        label_file = os.path.join(label_dir, s+f"{frame_num:06d}.txt")
        with open(label_file, "a+") as f:
            img_path = os.path.join(img_dir, f"{frame_num:06d}.jpg")
            img_width, img_height = Image.open(img_path).size
            if(img_width!=1920 or img_height!=1080):
                continue
            x_center = (x + w / 2) / 1920
            y_center = (y + h / 2) / 1080
            if(x_center<0 or y_center<0 or x_center>1 or y_center>1):
                continue
            if(h>img_height or w>img_width):
                continue
            img_out_path = os.path.join(img_dir_out, s+ f"{frame_num:06d}.jpg")
            # copy the image to the output folder
            img= Image.open(img_path)
            img.resize((1080, 1080))
            img.save(img_out_path)
            # normalize the coordinates
            # x_min = max(0, x)
            # y_min = max(0, y)
            # x_max = min(x + w, img_width)
            # y_max = min(y + h, img_height)
            # x_center = (x_min + x_max) / 2 / img_width
            # y_center = (y_min + y_max) / 2 / img_height
            # width_norm = (x_max - x_min) / img_width
            # height_norm = (y_max - y_min) / img_height
            # x_center = max(0, min(1, x_center))
            # y_center = max(0, min(1, y_center))
            # width_norm = max(0, min(1, width_norm))
            # height_norm = max(0, min(1, height_norm))
            # x_center= x+w/2
            # y_center= y+h/2
            # x_center = (x + w / 2) / 1920
            # y_center = (y + h / 2) / 1080
           
            width_norm= w/1920
            height_norm= h/1080
            freqs[label]+=1
            # write the label to the file
            # print("here" , num)
            num+=1
            f.write(f"{label} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")


# specify the path to the MOT16 dataset folder
mot16_folder = "/Users/tejasr/Documents/IITK/Semesters/sem 6/CS776/Project/MOT16/train/"

# loop over each sequence in the MOT16 dataset
for sequence_folder in os.listdir(mot16_folder):
    # construct the path to the gt.txt file and img1 folder
       # construct the path to the gt.txt file and img1 folder
    gt_file = os.path.join(mot16_folder, sequence_folder, "gt", "gt.txt")
    img_dir = os.path.join(mot16_folder, sequence_folder, "img1")
    # out_dir= os.path.join(mot16_folder, sequence_folder, "out")
    out_dir= os.path.join(mot16_folder, "outnewVal")
    s= sequence_folder+"_"
    if(sequence_folder==".DS_Store" or sequence_folder=="out1" or sequence_folder=="out" or sequence_folder=="out2" or sequence_folder=="outnew"):
        continue
    # convert the MOT16 format to YOLO format
    print(sequence_folder + " started")
    convert_to_yolo_format(gt_file, img_dir, out_dir)
    print(sequence_folder + " Completed")
    
print(freqs)