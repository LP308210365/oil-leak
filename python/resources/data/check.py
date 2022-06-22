import json5
"""
about filesystem
"""
import shutil
import os
import glob
"""
log
"""
import logging
"""
resources
"""
from resources.data import Abnormal

def check(logger, checker, user_settings, test):
  logger.debug('check type')

  checker.check(
    (
      'debug_level',
      'img_path', 'video_path',
      'language', 'vgg',
      'svm_model_path', 'vgg_model_path', 'xgboost_model_path',
      'model_t', 'generation_t',
    ), str
  )
  checker.check(
    (
      'frame_range',
      'init_box_scale',
    ), list
  )
  checker.check(
    (
      'data'
    ), dict
  )
  checker.check(
    (
      'file_output', 'time_debug',
      'linux', 'sift', 'OF', 'detectShadows',
      'Retina', 'debug_per_frame',
      'cuda',
    ), bool
  )
  checker.check(
    (
      'delay', 'height', 'interval',
      'fps', 'limit_size', 'app_fps',
      'max_iter', 'num_epochs', 'step_size',
      'batch_size', 'num_workers', 'nthread',
      'num_round',
    ), int
  )
  checker.check(
    (
      'compression_ratio', 'varThreshold',
      'learning_rate', 'momentum',
      'gamma',
      'init_box_scale.0.0',
      'init_box_scale.0.1',
      'init_box_scale.1.0',
      'init_box_scale.1.1',
    ), float
  )

  logger.debug('check legal')




  checker.check(
    (
      'resource_path',
      'img_path', 'video_path',
      'svm_model_path', 'vgg_model_path', 'xgboost_model_path',
      'language',
    ),
    checker.len_not, 0
  )
  checker.check('frame_range', checker.range)
  checker.check('debug_level', checker.within, ('debug', 'info', 'warn', 'error', 'critical'))
  checker.check('vgg', checker.within, (
    '11', '11bn', '13', '13bn',
    '16', '16bn', '19', '19bn',
  ))
  checker.check('model_t', checker.within, ('xgboost', 'vgg', 'svm', 'none'))
  checker.check('generation_t', checker.within, ('video', 'image'))
  checker.check('max_iter', checker.plus_or_minus1)
  checker.check('num_workers', checker.plus_or_zero)
  checker.check(
    (
      'delay', 'height',
      'interval', 'fps',
      'limit_size', 'compression_ratio',
      'app_fps', 'varThreshold',
      'num_epochs', 'learning_rate',
      'momentum', 'step_size', 'gamma',
      'batch_size', 'nthread', 'num_round',
    ), checker.plus
  )

  if checker.dirty:
    logger.error("参数检查失败。。。请调整后再运行")
    from sys import exit
    exit(1)

  logger.debug('将debug_level转换为logging.枚举类型')

  debug_level = user_settings['debug_level']
  user_settings['debug_level'] = {
    'debug':    logging.DEBUG,
    'info':     logging.INFO,
    'warn':     logging.WARN,
    'error':    logging.ERROR,
    'critical': logging.CRITICAL,
  }[debug_level]


  logger.debug('测试的情况下该返回了')
  if test:
    return user_settings

  logger.debug('清空输出文件夹')

  if user_settings['file_output']:
    img_path = user_settings['img_path']
    if os.path.exists(img_path):
      shutil.rmtree(img_path)
    os.mkdir(img_path)
    video_path = user_settings['video_path']

    logger.debug('img和video在一个路径下就不重复做了')

    if img_path != video_path:
      if os.path.exists(video_path):
        shutil.rmtree(video_path)
      os.mkdir(video_path)
  return user_settings 
