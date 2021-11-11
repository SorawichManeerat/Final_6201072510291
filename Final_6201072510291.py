import cv2 as cv
import numpy as np 

sift = cv.SIFT_create()

lk_params = dict( winSize = (12,11),
                  maxLevel = 10,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TermCriteria_COUNT, 1000, 0.001) )


ref_img = cv.imread("Templates/Template-4.png")
ref_img = cv.cvtColor(ref_img,cv.COLOR_BGR2RGB)
old_gray = cv.cvtColor(ref_img,cv.COLOR_RGB2GRAY)

ref_keypoing, ref_des = sift.detectAndCompute(old_gray, None)

search_img = cv.VideoCapture('Dataset-1/left_output-1.avi')

success,img2 = search_img.read()
imgOg = img2.copy()
img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

search_keypoing, search_des = sift.detectAndCompute(img2, None)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(check=50)

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(ref_des,search_des,k=2)

matchMask = [[0,0] for i in range(len(matches))]

good_match = list()
good_match_list = list()
for m,n in matches :
    if m.distance < 0.7*n.distance :
        good_match.append(m)
        good_match_list.append([m])


MIN_MATCH_COUNT = 10
if len(good_match) > MIN_MATCH_COUNT :
    src_pts = np.float32([ ref_keypoing[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
    dst_pts = np.float32([ search_keypoing[m.trainIdx].pt for m in good_match ]).reshape(-1,1,2)
    
    
    
while True :
    ret,frame = search_img.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    


    p1, st, err = cv.calcOpticalFlowPyrLK(img2, frame_gray, dst_pts, None,**lk_params)   
    

    M, mask1 = cv.findHomography(src_pts, p1, cv.RANSAC,0.0004)
    h,w = ref_img.shape[:2]
  
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    
    img = frame
    cv.polylines(img,[np.int32(dst)],True,(195,9,15),3,cv.LINE_AA)
  
    cv.imshow('Crowd heatmapping with optical flow',img)
    k = cv.waitKey(1) & 0xff
    if k == 27 :
        break

    img2 = frame_gray.copy()
    dst_pts = p1.reshape(-1,1,2)
       

cv.destroyAllWindows()
cap.release
