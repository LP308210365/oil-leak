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
      'path', 'debug_level',
      'img_path', 'video_path',
      'language', 'vgg',
      'svm_model_path', 'vgg_model_path', 'xgboost_model_path',
      'model_t', 'generation_t',
    ), str
  )
  checker.check(
    (
      'videos', 'frame_range',
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

  class_info_exists = checker.check('path', checker.has_file, 'class_info_file')
  checker.check(
    (
      'videos', 'resource_path',
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
  checker.check('data', checker.len_is, 2)
  checker.check('data.train', checker.is_dir)
  checker.check('data.test', checker.is_dir)
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

  logger.debug('??????class_info????????????')

  if class_info_exists:
    class_info_file = f"{user_settings['path']}/{user_settings['class_info_file']}"
    with open(class_info_file, encoding='utf-8') as f:
      user_settings['class_info'] = json5.load(f)
    checker.check('class_info', checker.within, Abnormal.Abnormal.names())

  if checker.dirty:
    logger.error("????????????????????????????????????????????????")
    from sys import exit
    exit(1)

  logger.debug('???debug_level?????????logging.????????????')

  debug_level = user_settings['debug_level']
  user_settings['debug_level'] = {
    'debug':    logging.DEBUG,
    'info':     logging.INFO,
    'warn':     logging.WARN,
    'error':    logging.ERROR,
    'critical': logging.CRITICAL,
  }[debug_level]

  logger.debug("?????????????????????'/'")

  path = user_settings['path']
  if path[-1] == '/':
    path = path[:-1]
    user_settings['path'] = path

  logger.debug('???????????????????????????????????????????????????videos')

  videos = user_settings['videos']

  __videos = []
  class_info = user_settings['class_info'].keys()
  for video in videos:
    for f in glob.glob(f'{path}/{video}'):
      f = f.replace('\\', '/')
      splits = f.split('.')
      if len(splits) <= 1:
        logger.debug('?????????????????????')
        continue
      if splits[-1] == 'json':
        logger.debug('????????????')
        continue
      name = splits[-2].split('/')[-1]
      if len(name) == 0:
        logger.debug('??????????????????')
        continue
      if name not in class_info:
        logger.debug(f'???????????????????????????"{name}"')
        continue
      __videos.append((name, f))

  user_settings['videos'] = __videos

  logger.debug('??????????????????????????????')
  if test:
    return user_settings

  logger.debug('?????????????????????')

  if user_settings['file_output']:
    img_path = user_settings['img_path']
    if os.path.exists(img_path):
      shutil.rmtree(img_path)
    os.mkdir(img_path)
    video_path = user_settings['video_path']

    logger.debug('img???video????????????????????????????????????')

    if img_path != video_path:
      if os.path.exists(video_path):
        shutil.rmtree(video_path)
      os.mkdir(video_path)
  return user_settings 
