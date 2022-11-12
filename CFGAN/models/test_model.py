import torch
from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks


# class TestModel(BaseModel):
#     def name(self):
#         return 'TestModel'
#
#     def __init__(self, opt):
#         assert(not opt.isTrain)
#         super(TestModel, self).__init__(opt)
#         self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
#
#         self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
#                                       opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
#                                       opt.learn_residual)
#
#         self.load_network(self.netG, 'G', opt.which_epoch)
#
#         print('---------- Networks initialized -------------')
#         networks.print_network(self.netG)
#         print('-----------------------------------------------')
#
#     def set_input(self, input):
#         # we need to use single_dataset mode
#         inputA = input['A']
#         self.input_A.resize_(inputA.size()).copy_(inputA)
#         self.image_paths = input['A_paths']
#
#     def test(self):
#         with torch.no_grad():
#             self.real_A = Variable(self.input_A)
#             self.fake_B = self.netG.forward(self.real_A)
#
#     # get image paths
#     def get_image_paths(self):
#         return self.image_paths
#
#     def get_current_visuals(self):
#         real_A = util.tensor2im(self.real_A.data)
#         fake_B = util.tensor2im(self.fake_B.data)
#         return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert (not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)

        self.load_network(self.netG, 'G', opt.which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        inputA = input['A']
        inputB = input['B']
        self.input_A.resize_(inputA.size()).copy_(inputA)
        self.input_B.resize_(inputB.size()).copy_(inputB)
        self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)
            self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])