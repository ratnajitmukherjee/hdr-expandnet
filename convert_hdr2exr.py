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

Copyright (c) 2018-Ratnajit Mukherjee, All rights reserved.
***********************************************************************************************************************
"""

"""
*********************************NOTE: Ratnajit Mukherjee**************************************************************
The following are the jobs of this script:
1) Convert HDR images to EXR images 
2) Scale the HDR image within a particular range input by user for example: [0.0005, 0.99]. Primarily created to scale
the data required for other network training purposes but can be used for other purposes.
***********************************************************************************************************************
"""
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import argparse
import os
import cv2
import OpenEXR
import Imath


class ConvertHDR2EXR:
    def __init__(self):
        print("\n STARTING HDR TO EXR CONVERSION. USING ALL CPU CORES\n")

    def write_exr(self, hdr, outfilename):
        """
        Write EXR image (half channel header)
        :param hdr: input HDR in opencv BGR format
        :param outfilename: outputfile name in exr
        :return: None
        """
        header = OpenEXR.Header(hdr.shape[1], hdr.shape[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])

        out = OpenEXR.OutputFile(outfilename, header)
        # NOTE: OpenCV reads as BGR so change color indexing as RGB before writing
        r_channel = (hdr[:, :, 2]).astype(np.float16).tostring()
        g_channel = (hdr[:, :, 1]).astype(np.float16).tostring()
        b_channel = (hdr[:, :, 0]).astype(np.float16).tostring()
        out.writePixels({'R': r_channel, 'G': g_channel, 'B': b_channel})
        return

    def process_hdr(self, filename, scale_min, scale_max, output_dir):
        hdr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # this must be read as unchanged else OpenCV reads incorrectly
        hdr = np.clip(hdr, 1e-9, max(hdr.ravel()))  # ensuring that there is scaling (although might not be required)

        if scale_min is not None and scale_max is not None:
            hdr_scaled = ((hdr - min(hdr.ravel()) * (scale_max - scale_min)) / (max(hdr.ravel()) - min(hdr.ravel()))) \
                         + scale_min    # scale the HDR within given range
        else:
            hdr_scaled = hdr

        # manipulation of the filename to remove _prediction
        f_name = os.path.basename(filename)
        basename, extension = f_name.split('_prediction')
        out_name = os.path.join(output_dir, basename + '.exr')

        self.write_exr(hdr=hdr_scaled, outfilename=out_name)
        print("\n File {0} written to disk.".format(basename + '.exr'))
        return

    def process_filelist(self, filelist, scale_min, scale_max, output_dir):
        """
        Processing the hdr filelist in an embarrasingly parallel way
        """
        # iterating through the filelist
        nCPU = multiprocessing.cpu_count()
        Parallel(n_jobs=nCPU)(delayed(self.process_hdr)(filelist[index], scale_min, scale_max, output_dir)
                              for index in range(0, len(filelist)))


    def get_file_list(self, input_dir):
        """
        Function to get the filelist from the root folder
        :param input_dir: root directory full path
        :return: Path of all .hdr files in the root directory
        """
        filelist = [os.path.join(dirname, filename) for dirname, subdirs, filenames in os.walk(input_dir)
                    for filename in filenames if filename.endswith('.hdr')]
        return filelist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="The input directory where the HDR (.hdr) files are stored",
                        type=str, required=True)
    parser.add_argument("-o", "--output_dir", help="The output directory where the HDR (.exr) files will be stored",
                        type=str, required=True)
    parser.add_argument("-min", "--scale_min", help="The minimum value to which the (.exr) images will be scaled",
                        type=float)
    parser.add_argument("-max", "--scale_max", help="The maximum value to which the (.exr) images will be scaled",
                        type=float)
    args = parser.parse_args()

    # processing the input arguments
    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.scale_min is not None:
        scale_min = args.scale_min
    else:
        scale_min = None

    if args.scale_max is not None:
        scale_max = args.scale_max
    else:
        scale_max = None

    # Calling the class functions
    hdr2exr = ConvertHDR2EXR()
    hdr_filelist = hdr2exr.get_file_list(input_dir=input_dir)
    hdr2exr.process_filelist(filelist=hdr_filelist, scale_min=scale_min, scale_max=scale_max, output_dir=output_dir)

    # Ending the process
    print("\n PROCESS COMPLETE. TOTAL {0} FILES PROCESSED".format(len(hdr_filelist)))