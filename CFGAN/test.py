import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from util.metrics import SSIM
from util.metrics import ColorDistanceMean
from PIL import Image
import os
import numpy as np
import cv2
from CSE import CSE
opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.fineSize = 256
opt.dataroot = ''
opt.dataset_mode = 'aligned'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
avgPSNR = 0.0
avgSSIM = 0.0
avgColorDistance = 0.0
counter = 0

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    counter = i
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    avgPSNR += PSNR(visuals['Restored_Train'], visuals['Sharp_Train'])
    avgSSIM += SSIM(visuals['Restored_Train'], visuals['Sharp_Train'])
    avgColorDistance += CSE(visuals['Restored_Train'], visuals['Sharp_Train'])
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)
avgPSNR /= counter
avgSSIM /= counter
avgColorDistance /= counter
print('PSNR = %f, SSIM = %f,avgColorDistance = %f' %
      (avgPSNR, avgSSIM,avgColorDistance))

webpage.save()
