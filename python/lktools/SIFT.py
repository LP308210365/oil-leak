import cv2
import numpy as np
import lktools.Timer

def sift_kp(image):
  """

  :param image: 输入图像
  :return:      画了关键点的图片，关键点，关键点描述
  """
  gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  sift=cv2.xfeatures2d.SIFT_create()
  kp,des = sift.detectAndCompute(image,None)
  kp_image = cv2.drawKeypoints(gray_image,kp,None)
  return kp_image,kp,des
 
def get_good_match(des1,des2):
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1, des2, k=2) #des1为模板图，des2为匹配图
  matches = sorted(matches,key=lambda x:x[0].distance/x[1].distance)
  good = []
  for m, n in matches:
    if m.distance < 0.75 * n.distance:
      good.append(m)
  return good

@lktools.Timer.timer_decorator()
def siftImageAlignment(img1,img2):
  _,kp1,des1 = sift_kp(img1)
  _,kp2,des2 = sift_kp(img2)
  goodMatch = get_good_match(des1,des2) #按距离排序，保留前几个特征匹配点

  #TODO 若透视变换失败，则用原图
  imgOut = img2
  if len(goodMatch) > 4:
    #ptsA,ptsB为排序好的几个特征匹配点
    ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)   #statu项为0，表示

    # TODO H矩阵可能出现计算失败的情况,即status都为0
    if 1 in status:
      imgOut = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
  return imgOut

def siftImageDraw(img1,img2):
  _,kp1,des1 = sift_kp(img1)
  _,kp2,des2 = sift_kp(img2)
  goodMatch = get_good_match(des1,des2)
  
  return cv2.drawMatches(img1, kp1, img2, kp2, goodMatch[:5], None, flags=2)
  #----or----
  #goodMatch = np.expand_dims(goodMatch,1)
  #return cv2.drawMatchesKnn(img1, kp1, img2, kp2, goodMatch[:5], None, flags=2)