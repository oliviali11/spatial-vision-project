import glob
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import math

# main function that grabs data + creates graphs
"""
  Inputs:
    depth_model - Choose depth model to grab data from, current options include 'depth_anywhere', 'midas'.
    object_movement - Includes main object + camera movement type. Ex: 'car_panover', 'garfield_rotate'
    weight - This is the weight the MiDaS model uses. Defaulted to '-dpt_beit_base_384.pfm'. If you want to use
      a different weight, go to MiDaS' github & download the weight.
    graph - If True, displays the graph. Defaulted to display graph.
    compare - (Optional) Array of items to compare + make ratio of
      Ex: [[ob1_a, ob2_a], [obj1_b, obj2_b], ...] will plot obj1_a/obj2_a and obj1_b/obj2_b
"""
def depth_process(depth_model, object_movement, weight="-dpt_beit_base_384.pfm", graph = True, compare = []):
      annotated_file_path = f"depth_anywhere/annotated/{object_movement}/{object_movement}_annotated.xml"
      depth_path = f"{depth_model}/depth_array/{object_movement}/0"
      depth_fend = "_depth.npy"
      camPos_file = f"depth_anywhere/cam_position/{object_movement}_camposition.txt"
      real_blender = []
      

      # Only 1 main object; normalize values off middle frame
      if compare == []:
        # all_bbox - grabs all bounding boxes for each corresponding label
        # labels - returns array of all location labels (Ex: ['pommel', 'blade', ...])
        (all_bbox, labels) = process_annotations(annotated_file_path, object_movement)
        result = []

        if depth_model == 'midas':
          for i in range(len(all_bbox)):
            depth_path = f"midas/midas/pfm_file/{object_movement}/0"
            depth_fend = weight
            # depth_mean_midas returns an array of mean pixel values within each bounding box
            result_mean = depth_mean_midas(all_bbox[i][1:], depth_path, depth_fend)[0]
            result.append(result_mean)
        
        elif depth_model == 'depth_anywhere':
          for i in range(len(all_bbox)):
              # depth_median_mean returns an array of mean and median pixel values within each bounding box
              # mean: depth_median_mean(...)[0], median: depth_median_mean(...)[1]
              result_mean = depth_median_mean(all_bbox[i][1:], depth_path, depth_fend)[0]
              result.append(result_mean)
          
        # Grabs real blender distances from camera to each of the label for every frame
        # Returns array of distances per frame for each label
        for j in range(len(labels)):
            objPos_file = f"depth_anywhere/target_location/{object_movement}/{object_movement}_{labels[j]}.txt"
            real_blender.append(blenderDist(camPos_file, objPos_file))
        
        # If graph is True, display a graph!
        if graph:
            title = depth_model + " " + object_movement
            graph_plot(result, real_blender, title, labels)
     
      # multiple main objects to compare
      else:
         # Returns bounding box for object 1, object 2
         # Ex: If scene is car_cone_panover, it returns an array of bounding boxes for both car & cone
         (bbox1, bbox2) = process_multiple_annotations(annotated_file_path, object_movement, compare)
         result, normalizedValues = [], []
         result_mean1, result_mean2 = [], []
         compnames = []
        
        # gather mean values for both main objects adding to result_mean1/2 seperately so we can compare them
         if depth_model == 'depth_anywhere':
          for i in range(len(bbox1)):
              result_mean1.append(depth_median_mean(bbox1[i][1:], depth_path, depth_fend)[0])
              result_mean2.append(depth_median_mean(bbox2[i][1:], depth_path, depth_fend)[0])
              compareNames = bbox1[i][0] + '-' + bbox2[i][0]
              compnames.append(compareNames)
         elif depth_model == 'midas':
            depth_path = f"midas/midas/pfm_file/{object_movement}/0"
            depth_fend = weight
            for i in range(len(bbox1)):
              result_mean1.append(depth_mean_midas(bbox1[i][1:], depth_path, depth_fend)[0])
              result_mean2.append(depth_mean_midas(bbox2[i][1:], depth_path, depth_fend)[0])
              compareNames = bbox1[i][0] + '-' + bbox2[i][0]
              compnames.append(compareNames)

         # creates newArr which is normalized value of obj1/obj2
         # if any value is 0, use the other corresponding object's value
         for j in range(len(result_mean1)):
            newArr = []
            min_len = min(len(result_mean1[j]), len(result_mean2[j]))
            for k in range(min_len):
              if result_mean1[j][k] == 0 and result_mean2[j][k] == 0:
                  newArr.append(1)
              elif result_mean1[j][k] == 0:
                  newArr.append(result_mean2[j][k])
              elif result_mean2[j][k] == 0:
                  newArr.append(result_mean1[j][k]) 
              else:
                  newArr.append(result_mean1[j][k]/result_mean2[j][k])
            normalizedValues.append(newArr)

         # gather real distance data for both main objects
         for k in range(len(compare)):
            obj1, obj2, movement = object_movement.split("_")
            obj1Pos_file = f"depth_anywhere/target_location/{object_movement}/{obj1}_{compare[k][0]}.txt"
            obj2Pos_file = f"depth_anywhere/target_location/{object_movement}/{obj2}_{compare[k][1]}.txt"

            real_blender.append(ratioblenderDist(camPos_file, obj1Pos_file, obj2Pos_file))
         title = f"Depth Ratio for {object_movement}"

         graph_plot(normalizedValues, real_blender, title, compnames, "Depth Ratio")
      return result

# plot data on graph
def graph_plot(data, real_blender, graph_title, labels, y_axis = 'Normalized Depth'):
    plt.title(graph_title)
    plt.xlabel('Frame')
    plt.ylabel(y_axis)

    # Currently, only has data from every 5 frames
    x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    colors = ['r','orange','y','g','c','b','m','purple', 'brown','k']

    for i in range(len(data)):
        plt.plot(x[:len(data[i])], data[i], colors[i], label=labels[i])
    for j in range(len(real_blender)):
        plt.plot(real_blender[j], colors[j], linestyle='--')

    plt.legend()
    plt.show()

# Reads in xml file from specified path
def process_annotations(annotated_file_path, object_movement):
    tree = ET.parse(annotated_file_path)
    root = tree.getroot()

    pattern = f"depth_anywhere/target_location/{object_movement}/{object_movement}_*.txt"
    matchFiles = glob.glob(pattern)

    # grabLabels gets all labels we want in array ['blade', 'pommel', ...]
    grabLabels = [os.path.basename(file).replace(f"{object_movement}_", "").replace(".txt", "") for file in matchFiles]
    labelsMap = {} # labels : index
    bboxes = []    # [[x1, y1, x2, y2], [...], ...]
    
    for i in range(len(grabLabels)):
        labelsMap[grabLabels[i]] = i
        bboxes.append([grabLabels[i]])

    # Iterate over each <image> element
    for image in root.findall('image'):
        image_name = image.get('name')

        # Iterate over each <box> element within the <image>
        for box in image.findall('box'):
            label = box.get('label')
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))

            idx = labelsMap[label]

            bboxes[idx].append([image_name[1:4], [xtl + 1, ytl + 1, xbr, ybr]])
    return (bboxes, grabLabels)

def process_multiple_annotations(annotated_file_path, object_movement, compare):
    # compare = [[location_1a, location_1b], [location_2a, location_2b], ...]
    tree = ET.parse(annotated_file_path)
    root = tree.getroot()
    firstobject, secondobject, temp = object_movement.split("_")

    # Ex: pattern1 = "depth_anywhere/target_location/car_cone_panover/car_*.txt"
    pattern1 = f"depth_anywhere/target_location/{object_movement}/{firstobject}_*.txt"
    # matchFiles1 = glob.glob(pattern1)

    # Ex: pattern2 = "depth_anywhere/target_location/car_cone_panover/cone_*.txt"
    pattern2 = f"depth_anywhere/target_location/{object_movement}/{secondobject}_*.txt"
    # matchFiles2 = glob.glob(pattern2)

    # grabLabels gets all labels we want in array ['blade', 'pommel', ...]

    # grabLabels = [os.path.basename(file).replace(f"{object_movement}_", "").replace(".txt", "") for file in matchFiles]
    object1_Map = {} # labels : index
    object1_bboxes = []    # [[x1, y1, x2, y2], [...], ...]

    object2_Map = {} # labels : index
    object2_bboxes = []    # [[x1, y1, x2, y2], [...], ...]
    
    for i in range(len(compare)):
        object1_Map[compare[i][0]] = i
        object2_Map[compare[i][1]] = i
        object1_bboxes.append([compare[i][0]])
        object2_bboxes.append([compare[i][1]])


    # Iterate over each <image> element
    for image in root.findall('image'):
        image_name = image.get('name')

        # Iterate over each <box> element within the <image>
        for box in image.findall('box'):
            label = box.get('label')
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))

            if label in object1_Map:
              idx = object1_Map[label]
              object1_bboxes[idx].append([image_name[1:4], [xtl + 1, ytl + 1, xbr, ybr]])
            elif label in object2_Map:
              idx = object2_Map[label]
              object2_bboxes[idx].append([image_name[1:4], [xtl + 1, ytl + 1, xbr, ybr]])
    return(object1_bboxes, object2_bboxes)

def depth_median_mean(bbox, depth_path, depth_fend):
  start_index = '001'
  size_bbox = len(bbox)
  frameRef = size_bbox // 2  # normalize off middle frame

  # code to load the depth image
  dimg = np.load(depth_path + start_index + depth_fend)

  depth_mean = [0] * size_bbox
  depth_median = [0] * size_bbox

  for i in range(size_bbox):
      index = bbox[i][0]
      dimg = np.load(depth_path + index + depth_fend)

      # grab array of pixel values in each bounding box
      i_coords = bbox[i][1]
      depth_patch = dimg[i_coords[0]:i_coords[2],i_coords[1]:i_coords[3]]

      # calculate mean of pixel values within each box/patch
      depth_mean[i] = np.mean(depth_patch)
      depth_median[i] = np.median(depth_patch)

  normalized_mean = [x/depth_mean[frameRef] for x in depth_mean]
  normalized_median = [x/depth_median[frameRef] for x in depth_median]
  return (normalized_mean, normalized_median)

def depth_mean_midas(bbox, depth_path, depth_fend):
  start_index = '001'
  size_bbox = len(bbox)
  frameRef = size_bbox // 2

  depth_mean = [0] * size_bbox
  depth_median = [0] * size_bbox

  for i in range(size_bbox):
      index = bbox[i][0]
      
      dimg = read_pfm(depth_path + index + depth_fend)
    
      i_coords = bbox[i][1]
      depth_patch = dimg[i_coords[0]:i_coords[2],i_coords[1]:i_coords[3]]

      depth_mean[i] = np.mean(depth_patch)
      depth_median[i] = np.median(depth_patch)


  if depth_mean[frameRef] == 0:
     if depth_median[frameRef] != 0:
        print("Setting depth_mean to median to accomodate 0 depth value")
        depth_mean[frameRef] = depth_median[frameRef]
     else:
        print("Setting frame ref mean to 1 since mean + median are both 0")
        depth_mean[frameRef] = 1


  normalized_mean = [x/depth_mean[frameRef] for x in depth_mean]
  # normalized_median = [x/depth_median[frameRef] for x in depth_median]
  normalized_median = []

  return (normalized_mean, normalized_median)


############################################# Real Blender Positions ####################################################################
# multiDimenDist Code from https://stackoverflow.com/questions/51272288/how-to-calculate-the-vector-from-two-points-in-3d-with-python
def multiDimenDist(point1,point2):
   # find the difference between the two points
   deltaVals = [point2[dimension]-point1[dimension] for dimension in range(len(point1))]
   runningSquared = 0
   # because the pythagarom theorem works for any dimension we can just use that
   for coOrd in deltaVals:
       runningSquared += coOrd**2
   return runningSquared**(1/2)

def findVec(point1,point2,unitSphere = False):
  #setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
  finalVector = [0 for coOrd in point1]
  for dimension, coOrd in enumerate(point1):
      #finding total differnce for that co-ordinate(x,y,z...)
      deltaCoOrd = point2[dimension]-coOrd
      #adding total difference
      finalVector[dimension] = deltaCoOrd
  if unitSphere:
      totalDist = multiDimenDist(point1,point2)
      unitVector =[]
      for dimen in finalVector:
          unitVector.append( dimen/totalDist)
      return unitVector
  else:
      return finalVector

# Find real distance between vector camera -> object per frame
def magnitude(vector):
  return math.sqrt(sum(pow(element, 2) for element in vector))

# grabs camera translation from txtfile and returns [x,y,z]
def makeCameraTranslation(txtfile):
  # read in file's content
  file = open(txtfile, "r")
  contentArr = []

  while True:
    content = file.readline()
    contentArr.append(content.split("#"))
    if not content:
      break

  # make Camera Translation matrix
  camTranslation = []

  # split 3 x,y,z values for translation & rotation
  for i in range(1, len(contentArr) - 1):
    camTranslation.append((contentArr[i][1].split(",")))

  for j in range(len(camTranslation)):
    for m in range(3):
      camTranslation[j][m] = float(camTranslation[j][m])

  return camTranslation

def normalizeData(data, frameRef):
  result = [x/data[frameRef] for x in data]
  return result

# Given position of camera + object in Blender world, we can calculate the distance between the 2 coordinates
def blenderDist(camPos_file, objPos_file):
  camPosition = makeCameraTranslation(camPos_file)
  objPosition = makeCameraTranslation(objPos_file)

  blenderDist = []

  for i in range(len(camPosition)):
    blenderDist.append(magnitude(findVec(camPosition[i], objPosition[i])))
  # Normalizing data based on Frame 50 out of 100
  norm_blender = normalizeData(blenderDist, 50)
  return norm_blender


# Given position of camera + object in Blender world, we can calculate the distance between the 2 coordinates
def ratioblenderDist(camPos_file, obj1Pos_file, obj2Pos_file):
  camPosition = makeCameraTranslation(camPos_file)
  obj1Position = makeCameraTranslation(obj1Pos_file)
  obj2Position = makeCameraTranslation(obj2Pos_file)

  blenderDist = []

  for i in range(len(camPosition)):
    blenderDist.append(magnitude(findVec(camPosition[i], obj1Position[i])) / magnitude(findVec(camPosition[i], obj2Position[i])))
  # Normalizing data based on Frame 50 out of 100
  # norm_blender = normalizeData(blenderDist, 50)
  return blenderDist


# Custom function to read PFM files
def read_pfm(file_path):
    with open(file_path, 'rb') as file:
        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_line = file.readline().decode('utf-8').rstrip()
        dimensions = [int(i) for i in dim_line.split()]
        scale = float(file.readline().decode('utf-8').rstrip())
        data = np.fromfile(file, '<f' if scale < 0 else '>f')
        shape = (dimensions[1], dimensions[0], 3) if color else (dimensions[1], dimensions[0])

        return np.flipud(np.reshape(data, shape))