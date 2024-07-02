import glob
import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from pypfm import PFMLoader



def depth_process(depth_model, object_movement, weight="-dpt_beit_base_384.pfm", graph = True):
      annotated_file_path = f"depth_anywhere/annotated/{object_movement}/{object_movement}_annotated.xml"
      depth_path = f"depth_anywhere/depth_array/{object_movement}/0"
      depth_fend = "_depth.npy"
      camPos_file = f"depth_anywhere/cam_position/{object_movement}_camposition.txt"

      (all_bbox, labels) = process_annotations(annotated_file_path, object_movement)

      result = []
      real_blender = []

      if depth_model == 'midas':
        for i in range(len(all_bbox)):
          depth_path = f"midas/midas/pfm_file/{object_movement}/0"
          depth_fend = weight
          result_mean = depth_mean_midas(all_bbox[i][1:], depth_path, depth_fend)[0]
          result.append(result_mean)
      
      elif depth_model == 'depth_anywhere':
        for i in range(len(all_bbox)):
            result_mean = depth_median_mean(all_bbox[i][1:], depth_path, depth_fend)[0]        # result_median = depth_median_mean(all_bbox[i][1:], depth_path, depth_fend)[1]
            result.append(result_mean)
        
      for j in range(len(labels)):
          objPos_file = f"depth_anywhere/target_location/{object_movement}/{object_movement}_{labels[j]}.txt"
          real_blender.append(blenderDist(camPos_file, objPos_file))
      
      if graph:
          title = depth_model + " " + object_movement
          graph_plot(result, real_blender, title, labels)

      return result

def graph_plot(data, real_blender, graph_title, labels):
    plt.title(graph_title)
    plt.xlabel('Frame')
    plt.ylabel('Normalized Depth')
    x = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    colors = ['r','y','g','c','m','b','k', 'orange', 'purple', 'brown']

    for i in range(len(data)):
        plt.plot(x[:len(data[i])], data[i], colors[i], label=labels[i])
    for j in range(len(real_blender)):
        plt.plot(real_blender[j], colors[j], linestyle='--')

    plt.legend()
    plt.show()

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

def depth_median_mean(bbox, depth_path, depth_fend):
  
  start_index = '001'
  size_bbox = len(bbox)
  frameRef = size_bbox // 2

  # code to load the depth image
  dimg = np.load(depth_path + start_index + depth_fend)

  depth_mean = [0] * size_bbox
  depth_median = [0] * size_bbox

  for i in range(size_bbox):
      index = bbox[i][0]
      dimg = np.load(depth_path + index + depth_fend)

      i_coords = bbox[i][1]
      depth_patch = dimg[i_coords[0]:i_coords[2],i_coords[1]:i_coords[3]]

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
      # print(depth_path + index + depth_fend)
      
      dimg = read_pfm(depth_path + index + depth_fend)

      i_coords = bbox[i][1]
      depth_patch = dimg[i_coords[0]:i_coords[2],i_coords[1]:i_coords[3]]

      depth_mean[i] = np.mean(depth_patch)
      depth_median[i] = np.median(depth_patch)

  normalized_mean = [x/depth_mean[frameRef] for x in depth_mean]
  normalized_median = [x/depth_median[frameRef] for x in depth_median]

  return (normalized_mean, normalized_median)

def read_pfm(fileName):
  loader = PFMLoader(color=False, compress=False)
  pfm_data = loader.load_pfm(fileName)

  return pfm_data

############################################# Real Blender Positions ####################################################################
# multiDimenDist Code from https://stackoverflow.com/questions/51272288/how-to-calculate-the-vector-from-two-points-in-3d-with-python
def multiDimenDist(point1,point2):
   #find the difference between the two points, its really the same as below
   deltaVals = [point2[dimension]-point1[dimension] for dimension in range(len(point1))]
   runningSquared = 0
   #because the pythagarom theorm works for any dimension we can just use that
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

def read_pfm(fileName):
  loader = PFMLoader(color=False, compress=False)
  pfm_data = loader.load_pfm(fileName)

  return pfm_data

