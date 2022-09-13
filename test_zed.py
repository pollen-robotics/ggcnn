import pyzed.sl as sl
import cv2
from models.ggcnn2 import GGCNN2
from models.common import post_process_output
import torch
import numpy as np
from utils.dataset_processing import grasp
from cv2 import aruco

cam = sl.Camera()

init_params = sl.InitParameters()
init_params.depth_mode             = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units       = sl.UNIT.METER
init_params.camera_resolution      = sl.RESOLUTION.HD720
init_params.depth_minimum_distance = 0 # Set the minimum depth perception distance to 15cm
init_params.depth_maximum_distance = 1
init_params.camera_image_flip      = sl.FLIP_MODE.OFF  


# Open the camera
print("Opening the camera ...")
err = cam.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print(err)
    exit(1)
print("Camera successfully opened !")

runtime_parameters              = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

depth = sl.Mat()
_im    = sl.Mat()
im = None


model = GGCNN2()
# device = torch.device('cuda:0')
model.to("cpu")
model.load_state_dict(torch.load('models/ggcnn2_weights_cornell/epoch_50_cornell_statedict.pt'))
model.eval()

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_parameters =  aruco.DetectorParameters_create()

while True:
    crop = None
    cropDepth = None
    rects = None

    err = cam.grab(runtime_parameters)
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        cam.retrieve_image(_im, sl.VIEW.LEFT)
        im = _im.get_data()

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        depthMap = depth.get_data()
        depthMap = (depthMap*255).astype(np.uint8)
        point = (depthMap.shape[1]//2, depthMap.shape[0]-1)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_parameters)
        if len(corners) > 0:
            corner = corners[0][0]
            centerY = int((corner[0][1] + corner[2][1]) / 2)
            centerX = int((corner[0][0] + corner[2][0]) / 2)
            center = (centerX, centerY)

            objRect = (
                        (max(0, centerX-150), max(0, centerY-150)),
                        (min(centerX + 150, im.shape[1]), min(centerY+150, im.shape[0]))
                    )

            crop = im[objRect[0][1]:objRect[1][1], objRect[0][0]:objRect[1][0], :].copy()   
            cropDepth = depthMap[objRect[0][1]:objRect[1][1], objRect[0][0]:objRect[1][0]].copy()   

            im = cv2.rectangle(im, objRect[0], objRect[1], [0, 255, 0])

            im = cv2.circle(im, center, 20, (0, 255, 0), thickness=-1)

        if cropDepth is not None and crop is not None:
            with torch.no_grad():
                print(cropDepth.shape)
                print("========")
                d = np.clip((cropDepth - cropDepth.mean()), -1, 1)
                depthT = torch.from_numpy(d.reshape(
                    1, 1, 300, 300).astype(np.float32))
                b = model(depthT)
                # b = model(torch.Tensor(np.array([[cropDepth]])))    
                q_img, ang_img, width_img = post_process_output(b[0], b[1], b[2], b[3])
                grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                grs = grasp.GraspRectangles()
                for g in grasps : 
                    grs.append(g.as_gr)
                rects, _, _ = grs.draw(crop.shape, position=True)

                for g in grasps:
                    end_x = np.cos(g.angle)*g.length
                    end_y = np.sin(g.angle)*g.length
                    end = (end_x, end_y)
                    crop = cv2.line(crop, g.center, end, (255, 0, 0), thickness=5)

                    crop = cv2.circle(crop, g.center, 5, (0, 0, 255))

        if crop is not None and cropDepth is not None:
            cv2.imshow("crop", crop)
            cv2.imshow("cropDepth", cropDepth)
            cv2.imshow("rects", rects)
        cv2.imshow("depth", depthMap)
        cv2.imshow("im", im)
        cv2.waitKey(1)
