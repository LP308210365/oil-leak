"""
opencv
"""
import time

import cv2
"""
numpy
"""
import numpy
"""
lktools
"""
import lktools.LoggerFactory

import gc
import abc
import multiprocessing as mp
from multiprocessing import Process, Manager

logger = lktools.LoggerFactory.LoggerFactory('PreProcess').logger


class ABVideoCapture(abc.ABC):
  def __init__(self, cam, top=2):
    self.stack = Manager().list()
    self.queue = mp.Queue(maxsize=3)
    self.max_cache = top
    self.cam = cam
    self.write_process = Process(target=self.write)
    self.write_process.start()
    self.__read_gen = self.read_gen()
    self.timer = True

  @abc.abstractmethod
  def process_image(self, image):
    """对输入的图片进行处理并返回处理后的图片"""

  def read_gen(self):
    while True:
      if self.queue.qsize() != 0:
        img = self.process_image(self.queue.get())
        self.timer = True
        yield img
      else:
        if self.timer:
          start = time.time()
          self.timer = False
        end = time.time()
        if end - start > 3:
          yield None


  def read(self):
    try:
      frame = next(self.__read_gen)
      if frame is not None:
        return True, frame
      else:
        return False, None
      return True, next(self.__read_gen)
    except StopIteration:
      return False, None
    except TypeError:
      raise TypeError('{}.read_gen必须为生成器函数'.format(self.__class__.__name__))

  def __iter__(self):
    yield from self.__read_gen

  def release(self):
    self.write_process.terminate()

  def __del__(self):
    self.release()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.release()

  def write(self):
    """向共享缓冲栈中写入数据"""

    self.cap = cv2.VideoCapture(self.cam)
    while True:
      _, img = self.cap.read()
      if _:
        self.queue.put(img)
        # 每到一定容量清空一次缓冲栈
        # 利用gc库，手动清理内存垃圾，防止内存溢出
        if self.queue.qsize() >= self.max_cache:
          self.queue.get()
          gc.collect()


class VideoCapture(ABVideoCapture):
  def process_image(self, image):
    # 这里对图像的处理算法可以随意制定
    return image

def video_capture_size(path, height):
  capture = cv2.VideoCapture(path)
  if not capture.isOpened():
    logger.error(f'{path} not found')
    from sys import exit
    exit(1)
  m = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
  n = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
  fps = capture.get(cv2.CAP_PROP_FPS)
  count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
  scale = height / m
  m = height
  n = int(n * scale)
  capture.release()
  return m, n, fps, count

def subtraction(mat1, mat2, distance=0):
  """
  两帧做差（保证不溢出），把距离小于distance的置零，返回。
  """
  mat = numpy.abs(mat1.astype(numpy.int8) - mat2.astype(numpy.int8))
  if distance > 0:
    mat[mat < distance] = 0
  return mat.astype(numpy.uint8)

def min_max(rect):
  """
  将一个任意两点组成的rect，转换为(min, min), (max, max)的rect
  """
  if rect is None:
    return
  (x1, y1), (x2, y2), *_ = rect
  if x1 == x2 or y1 == y2:
    return
  if x1 < x2:
    x_min = x1
    x_max = x2
  else:
    x_min = x2
    x_max = x1
  if y1 < y2:
    y_min = y1
    y_max = y2
  else:
    y_min = y2
    y_max = y1
  return (x_min, y_min), (x_max, y_max)

def union_bounds(rects):
  """两框合并（最大化合并）"""
  (minx, miny), (maxx, maxy) = min_max(rects[0])
  for rect in rects[1:]:
    (x_min, y_min), (x_max, y_max) = min_max(rect)
    if x_min < minx:
      minx = x_min
    if y_min < miny:
      miny = y_min
    if x_max > maxx:
      maxx = x_max
    if y_max > maxy:
      maxy = y_max
  return (minx, miny), (maxx, maxy)

def matrix_within_rect(matrix, rect):
  """
  取出matrix中，rect部分的值
  """
  rect = min_max(rect)
  if rect is None:
    return
  (x1, y1), (x2, y2) = rect
  if x1 >= x2 or y1 >= y2:
    return
  return matrix[y1:y2, x1:x2]

def trim_to_rect(rect1, rect2):
  """
  将rect1缩到rect2内部，裁剪掉外部多余部分
  rect1与rect2为一个矩形的任意两个对角点
  """
  mm1 = min_max(rect1)
  if mm1 is None:
    return
  mm2 = min_max(rect2)
  if mm2 is None:
    return
  (x11, y11), (x12, y12) = mm1
  (x21, y21), (x22, y22) = mm2
  if x11 < x21:
    x11 = x21
  if x12 > x22:
    x12 = x22
  if y11 < y21:
    y11 = y21
  if y12 > y22:
    y12 = y22
  if x11 >= x12:
    return
  if y11 >= y12:
    return
  return (x11, y11), (x12, y12)

def rect_wh(rect):
  """
  计算一个矩形的宽高比
  rect为一个矩形的任意两个对角点
  """
  rect = min_max(rect)
  if rect is None:
    return
  (x0, y0), (x1, y1) = rect
  return (x1 - x0) / (y1 - y0)

def rect_size(rect):
  """
  计算一个矩形的面积
  rect为一个矩形的任意两个对角点
  """
  rect = min_max(rect)
  if rect is None:
    return
  (x0, y0), (x1, y1) = rect
  return (x1 - x0) * (y1 - y0)

def rect_center(rect):
  """
  计算一个矩形的中心
  rect为一个矩形的任意两个对角点
  """
  rect = min_max(rect)
  if rect is None:
    return
  (x0, y0), (x1, y1) = rect
  return (x1 - x0) // 2, (y1 - y0) // 2

def point_in_rect(point, rect):
  """
  判断一个点是否在一个矩形中
  rect为一个矩形的任意两个对角点

  若是，返回点在矩形中的相对坐标
  否则返回None
  """
  rect = min_max(rect)
  if rect is None:
    return
  leftdown, rightup = rect
  x, y = point
  if x < leftdown[0] or y < leftdown[1]:
    return
  if x > rightup[0] or y > rightup[1]:
    return
  return x - leftdown[0], y - leftdown[1]

def gray_to_bgr(img):
  """
  cv2.cvtColor(*, cv2.COLOR_GRAY2BGR) helper
  """
  if img is None:
    return
  return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def bgr_to_gray(img):
  """
  cv2.cvtColor(*, cv2.COLOR_BGR2GRAY) helper
  """
  if img is None:
    return
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bgr_to_hsv(img):
  """
  cv2.cvtColor(*, cv2.COLOR_BGR2HSV) helper
  """
  if img is None:
    return
  return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def draw(frame, rects):
  """
  将rects中的矩形绘制在frame上
  """
  img = frame.copy()
  for rect in rects.values():
    if rect:
      cv2.rectangle(img, *rect)
  return img
