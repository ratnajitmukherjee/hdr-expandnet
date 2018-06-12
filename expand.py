"""
***********************************************************************************************************************
BSD 3-Clause Clear License
Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Copyright (c) 2017-Demetris Marnerides, All rights reserved.
Modification by Ratnajit Mukherjee, 2018 with author's permission
***********************************************************************************************************************
"""

"""
************************************************************************************************************************
NOTE by Ratnajit Mukherjee

This main file expand.py has undergone minor modifications with the permission of the author. I claim no rights to the 
original code and please cite the author's work if you happen to use the code.

The rest of the project contains some files which are written by to process the predicted HDRs as EXRs (useful sometimes
to integrate into other projects. Also involves some clipping and scaling solely for the purpose of the my own work and 
does not effect the original code in any way.
************************************************************************************************************************
"""

import os
from os import path
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import cv2
from smooth import smoothen_luminance


def process_path(directory, create=False):
    directory = path.expanduser(directory)
    directory = path.normpath(directory)
    directory = path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = path.splitext(path.basename(directory))
    return path.dirname(directory), name, ext


# From torchnet
def compose(transforms):
    """Composes list of transforms (each accept and return one item)"""
    assert isinstance(transforms, list)
    for transform in transforms:
        assert callable(transform), "list of functions expected"

    def composition(obj):
        """Composite function"""
        for transform in transforms:
            obj = transform(obj)
        return obj
    return composition


def replace_specials_(x, val=0):
    x[np.isinf(x).sum() | np.isnan(x).sum()] = val
    return x


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)
    

def str2bool(x):
    if x is None or x.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        return True


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)]
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


def torch2cv(t_img):
    return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]


def resize(x):
    if opt.resize:
        return cv2.resize(x, (opt.width, opt.height))
    else:
        return x


# Model definition
class ExpandNet(nn.Module):
    def __init__(self):
        super(ExpandNet, self).__init__()

        def layer(nIn, nOut, k, s, p, d=1):
            return nn.Sequential(nn.Conv2d(nIn, nOut, k, s, p, d), 
                                 nn.SELU(inplace=True))
        self.nf = 64
        self.local_net = nn.Sequential(
            layer(3, 64, 3, 1, 1),
            layer(64, 128, 3, 1, 1),
        )

        self.mid_net = nn.Sequential(
            layer(3, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            nn.Conv2d(64, 64, 3, 1, 2, 2)
        )
        
        
        self.glob_net = nn.Sequential(
            layer(3, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            nn.Conv2d(64, 64, 4, 1, 0),
        )
        
        self.end_net = nn.Sequential(
            layer(256, 64, 1, 1, 0),
            nn.Conv2d(64, 3, 1, 1, 0),
            nn.Sigmoid()
        )

    # This uses stitching is for low memory usage
    def forward(self, t_input):
        vrs = torch.__version__.split('.')
        newer_version = False
        if int(vrs[0])==0 and int(vrs[1]) > 3:
            newer_version = True
            torch.no_grad()
        if t_input.dim() == 3:
            t_input = t_input.unsqueeze(0)
        if t_input.size(-3) == 1:
            # For grey images
            t_input = t_input.expand(1, 3, *t_input.size()[-2:])
        # Evaluate global features
        resized = cv2torch(cv2.resize(torch2cv(t_input.cpu()[0]), (256, 256)))
        resized = resized.unsqueeze(0)
        if opt.use_gpu:
            resized = resized.cuda()
        if newer_version:
            v_input_resize = Variable(resized)
        else:
            v_input_resize = Variable(resized, volatile=True)
        glob = self.glob_net(v_input_resize)

        overlap = 20 #
        skip = int(overlap/2)
        
        result = t_input.clone()
        if newer_version:
            v_input = Variable(t_input)
        else:
            v_input = Variable(t_input, volatile=True)
        v_input = torch.nn.functional.pad(v_input, (skip, skip, skip, skip))
        height, width = v_input.size(-2), v_input.size(-1)
        num_h = int(np.ceil((result.size(-2)-skip)/(opt.patch_size-overlap)))
        num_w = int(np.ceil((result.size(-1)-skip)/(opt.patch_size-overlap)))
        for h_index in range(num_h):
            for w_index in range(num_w):
                h_start = h_index*(opt.patch_size-overlap)
                w_start = w_index*(opt.patch_size-overlap)
                h_end = min(h_start + opt.patch_size, height)
                w_end = min(w_start + opt.patch_size, width)
                v_input_slice = v_input[:, :, h_start:h_end, w_start:w_end]
                loc = self.local_net(v_input_slice)
                mid = self.mid_net(v_input_slice)
                exp_glob = glob.expand(1, 64, h_end-h_start, w_end-w_start)
                fuse = torch.cat((loc, mid, exp_glob), 1)
                res = self.end_net(fuse).data
                # stitch
                h_start_stitch = h_index*(opt.patch_size-overlap)
                w_start_stitch = w_index*(opt.patch_size-overlap)
                h_end_stitch = min(h_start + opt.patch_size-overlap, height)
                w_end_stitch = min(w_start + opt.patch_size-overlap, width)
                res_slice = res[:,:,skip:-skip, skip:-skip]
                result[:,:,h_start_stitch:h_end_stitch, 
                           w_start_stitch:w_end_stitch].copy_(res_slice)
                del fuse, loc, mid, res
        return result[0]

"""
Parsing of input parameters
"""
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('ldr', nargs='+', type=process_path, help='Ldr image(s)')
arg('--out', type=lambda x: process_path(x, True),
    default=None, help='Output location.')
arg('--video', type=str2bool, default=False, help='Whether input is a video.')
arg('--patch_size', type=int, default=256,
    help='Patch size (to limit memory use).')
arg('--resize', type=str2bool, default=False, help='Use resized input.')
arg('--width', type=int, default=960, help='Image width resizing.')
arg('--height', type=int, default=540, help='Image height resizing.')
arg('--tag', default=None, help='Tag for outputs.')
arg('--use_gpu', type=str2bool, default=torch.cuda.is_available(),
    help='Use GPU for prediction.')
arg('--tone_map',
    choices=['exposure', 'reinhard', 'mantiuk', 'drago', 'durand'],
    default=None, help='Tone Map resulting HDR image.')
arg('--stops', type=float, default=0.0,
    help='Stops (loosely defined here) for exposure tone mapping.')
arg('--gamma', type=float, default=1.0,
    help='Gamma curve value (if tone mapping).')
opt = parser.parse_args()

net = ExpandNet()
net.load_state_dict(torch.load('weights.pth', map_location=lambda s, l: s))
net.eval()

preprocess = compose([
    lambda x: x.astype('float32'),
    resize,
    map_range,
    replace_specials_])

class Exposure(object):
    def __init__(self, stops, gamma):
        self.stops = stops
        self.gamma = gamma
    def process(self, img):
        return np.clip(img*(2**self.stops), 0, 1)**self.gamma

def tone_map(img, tmo_name):
    if (tmo_name == 'exposure'):
        tmo = Exposure(gamma=opt.gamma, stops=opt.stops)
    if (tmo_name == 'reinhard'):
        tmo = cv2.createTonemapReinhard(intensity=-1.0, 
                                        light_adapt=0.8, color_adapt=0.0)
    elif tmo_name == 'mantiuk':
        tmo = cv2.createTonemapMantiuk(saturation=1.0, scale=0.75)
    elif tmo_name == 'drago':
        tmo = cv2.createTonemapDrago(saturation=1.0, bias=0.85)
    elif tmo_name == 'durand':
        tmo = cv2.createTonemapDurand(contrast=3, saturation=1.0, 
                                      sigma_space=8, sigma_color=0.4)
    return tmo.process(img)


def create_name(inp, tag, ext, out, extra_tag):
    root, name, _ = split_path(inp)
    if extra_tag is not None:
        tag = '{0}_{1}'.format(tag, extra_tag)
    if out is not None:
        root = out
    return path.join(root, '{0}_{1}.{2}'.format(name, tag, ext))


# Option to create videos
if opt.video:
    if opt.tone_map is None:
        opt.tone_map = 'reinhard'
    video_file = opt.ldr[0]
    cap_in = cv2.VideoCapture(video_file)
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    n_frames = cap_in.get(cv2.CAP_PROP_FRAME_COUNT)
    lum_values = np.ndarray((2,int(n_frames)), dtype='float32')
    predictions = []
    lum_percs = []
    while cap_in.isOpened():
        perc = cap_in.get(cv2.CAP_PROP_POS_FRAMES)*100/n_frames
        print('\rConverting video: {0:.2f}%'.format(perc), end='')
        ret, loaded = cap_in.read()
        if loaded is None:
            break
        ldr_input = preprocess(loaded)
        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()
        predictions.append(torch2cv(net(t_input).cpu()))
        percs = np.percentile(predictions[-1], (1,25, 50, 75, 99))
        lum_percs.append(percs)
    print()
    cap_in.release()

    smooth_predictions = smoothen_luminance(predictions, lum_percs)
    
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out_vid_name = create_name(video_file, 'prediction', 'avi',
                               opt.out, opt.tag)
    out_vid = cv2.VideoWriter(out_vid_name, fourcc, fps, (width,height))
    for i, pred in enumerate(smooth_predictions):
        perc = (i+1)*100/n_frames
        print('\rWriting video: {0:.2f}%'.format(perc), end='')
        tmo_img = tone_map(pred, opt.tone_map)
        tmo_img = (tmo_img*255).astype(np.uint8)
        out_vid.write(tmo_img)
    print()
    out_vid.release()
else:
    for ldr_file in opt.ldr:
        loaded = cv2.imread(ldr_file,
                            flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        if loaded is None:
            print('Could not load {0}'.format(ldr_file))
            continue
        ldr_input = preprocess(loaded)
        if opt.resize:
            out_name = create_name(ldr_file, 'resized', 'jpg', opt.out,
                                   opt.tag)
            cv2.imwrite(out_name, (ldr_input*255).astype(int))
        
        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()

        """
        Changing this portion as a 0-1 range creates multiple issues during compression and color space changes
        as (divide by zero) condition pops up
        Change by Ratnajit Mukherjee, 2018 
        """
        prediction = map_range(torch2cv(net(t_input).cpu()), 1e-6, 1)

        out_name = create_name(ldr_file, 'prediction', 'hdr', opt.out,
                               opt.tag)
        cv2.imwrite(out_name, prediction)

        """
        Adding a simple print statement for monitoring purposes (easier while processing very large amounts of data
        Maybe will later add fancy progress bar
        Change by Ratnajit Mukherjee, 2018
        """
        path, basename = os.path.split(out_name)
        print("\n File {0} written to disk".format(basename))

        if opt.tone_map is not None:
            tmo_img = tone_map(prediction, opt.tone_map)
            out_name = create_name(ldr_file, 
                                   'prediction_{0}'.format(opt.tone_map),
                                   'jpg', opt.out, opt.tag)
            cv2.imwrite(out_name, (tmo_img*255).astype(int))