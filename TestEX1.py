import cv2 as cv
import numpy as np

params_dir = './new-camera_params/'

ret = np.load(params_dir+'ret.npy')
K = np.load(params_dir+'K.npy')
dist = np.load(params_dir+'dist.npy')
focal_length = (K[0,0]+K[1,1])/2

lk_params = dict(winSize=(12,11),
                maxLevel=10000,
                criteria=(cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT, 1000,0.001))

left_cap = cv.VideoCapture('Dataset-2/left_output.avi')
right_cap = cv.VideoCapture('Dataset-2/right_output.avi')

_, left_img = left_cap.read()
_, right_img = right_cap.read()

h,w = left_img.shape[:2]

win_size = 5
min_disparity = -1
max_disparity = 63
num_disparity = max_disparity - min_disparity

stereo = cv.StereoSGBM_create(minDisparity=min_disparity,
                            numDisparities=num_disparity,
                            blockSize=5,
                            uniquenessRatio=1,
                            speckleWindowSize=5,
                            speckleRange=5,
                            disp12MaxDiff=2,
                            P1=8*3*win_size**2,
                            P2=32*3*win_size**2)

new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(K,
                                                    dist,
                                                    (w,h),
                                                    1,
                                                    (w,h))


def downsampling_image(image,reduction_scale):
    for i in range(0, reduction_scale):
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv.pyrDown(image,dstsize=(col//2, row//2))
    return image

# Initiate SIFT detector
sift = cv.SIFT_create() #Features Detect

img = cv.imread('Templates/Template-4.png',cv.IMREAD_GRAYSCALE)
cap = cv.VideoCapture('Dataset-2/left_output.avi')

ret_, frame = cap.read()
frame_or = frame.copy()
gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# find the keypoints and descriptors wiht SIFT
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(gray_frame, None)

#features macthing
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(check=50) # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
matchMask = [[0,0] for i in range(len(matches))]

good = list()
good_ = list()
for m, n in matches:
    if m.distance < 0.70*n.distance:
        good.append(m)
        good_.append([m])


MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)


while True:

    _,left_img = left_cap.read()
    _,right_img = right_cap.read()

    left_img = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right_img = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

    left_img_undistorted = cv.undistort(left_img, K, dist, None, new_camera_matrix)
    right_img_undistorted = cv.undistort(right_img, K, dist, None, new_camera_matrix)
    
    left_img_down = downsampling_image(left_img_undistorted,2)
    right_img_down = downsampling_image(right_img_undistorted,2)

    disparity_map = stereo.compute(left_img_down, right_img_down)
    norm_disp = cv.normalize(disparity_map,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


    Q = np.float32([
        [1,0,0,-w/2.0],
        [0,1,0,-h/2.0],
        [0,0,0,float(-focal_length)],
        [0,0,-1/81.99,0]
    ])

    pointclouds = cv.reprojectImageTo3D(disparity_map, Q)


    ret, frame_ = cap.read()
    gray_frame_ = cv.cvtColor(frame_,cv.COLOR_BGR2GRAY)

    p1, st, err = cv.calcOpticalFlowPyrLK(gray_frame,
                                            gray_frame_,
                                            dst_pts,
                                            None,
                                            **lk_params)


    M, mask1 = cv.findHomography(src_pts, p1, cv.RANSAC, 3.3)#Matrix output

    #Perspective tranform
    h,w = img.shape[:2]
    pts = np.float32([[0,0],
                    [0,h-1],
                    [w-1,h-1],
                    [w-1,0]]
                    ).reshape(-1,1,2)

    dst = cv.perspectiveTransform(pts,M) #coordinate output

    det_frame = frame_

    dst_ = dst.ravel().tolist()
    #print(dst_)
    width = dst_[6]-dst_[0]
    high = dst_[3]-dst+[1]

    axis_x = np.int32(dst_[0]+width/2)
    axis_y = np.int32(dst_[1]+high/2)
    
    x_int = np.int32(axis_x/4)
    y_int = np.int32(axis_y/4)

    p = pointclouds[y_int][x_int]

    if (p[0] <= 0):
        p[0] = 0
    if (p[1] <= 0):
        p[1] = 0
    if (p[2] <= 0):
        p[2] = 0

    #print(axis_x,axis_y)
    #print(M)

    x_scale = np.float16(p[0]/100)
    y_scale = np.float16(p[1]/100)
    z_scale = np.float16(p[2]/100)

    print(x_scale, y_scale, z_scale)
    #print(p_)

    detail = '      X:'+str(x_scale)+',Y:'+str(y_scale)+',Z:'+str(z_scale)

    cv.polylines(det_frame, [np.int32(dst)], True,(255,0,255),3,cv.LINE_AA)
    cv.putText(det_frame, detail, (axis_x,axis_y),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,180,255),2)
    cv.imshow('gray_frame',det_frame)
    key = cv.waitKey(1)
    if key == 27:
        break
    gray_frame = gray_frame_.copy()
    dst_pts = p1.copy()


cap.release()
cv.destroyAllWindows()
