"""
opencv
"""
import cv2
"""
lktools
"""
import lktools.Timer
import lktools.LoggerFactory
from lktools.PreProcess import trim_to_rect

logger = lktools.LoggerFactory.LoggerFactory('FindObject').logger

def getContours(binary):
  if cv2.__version__.startswith('3'):
    _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  else:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

@lktools.Timer.timer_decorator()
def findObject(binary, rect):
  """对象框为红色"""
  logger.debug('计算图像中目标的轮廓')
  rects = []
  for c in getContours(binary):  #获得多个轮廓

    logger.debug('对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值')

    if cv2.contourArea(c) < 20:
      continue
    (x, y, w, h) = cv2.boundingRect(c) #注意：当有多个轮廓时，boundingRect只能获得最大的框

    logger.debug('该函数计算矩形的边界框')

    r = ((x, y), (x + w, y + h))
    r = trim_to_rect(r, rect)
    if r is None:
      continue
    rects.append((*r, (0, 0, 255), 2))
  return rects
