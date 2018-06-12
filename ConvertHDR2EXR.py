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
import numpy as np
import argparse
import os
import cv2

class ConvertHDR2EXR:
    def __init__(self):
        print("Starting to convert from HDR to EXR images. Using given CPU cores.")

    def write_exr(self, hdr, outfilename):
        pass

    def process_hdr(self, filename, scale_min, scale_max, output_dir):
        hdr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # this must be read as unchanged else OpenCV reads incorrectly
        hdr = np.clip(hdr, 1e-9, max(hdr.ravel()))  # ensuring that there is scaling (although might not be required)

        if scale_min is not None and scale_max is not None:
            hdr_scaled = ((hdr - min(hdr.ravel()) * (scale_max - scale_min)) / (max(hdr.ravel()) - min(hdr.ravel()))) \
                         + scale_min

        # manipulation of the filename to remove _prediction
        basename, extension = filename.split('_prediction')
        out_name = os.path.join(output_dir, basename + 'exr')

        self.write_exr(hdr=hdr_scaled, outfilename=out_name)
        print("\n File {0} written to disk.".format(basename + '.exr'))
        return

    def process_filelist(self, filelist):
        # need to write the code for parallel processing of files
        pass

    def get_file_list(self, input_dir):
        filelist = [os.path.join(dirname, filename) for dirname, subdirs, filenames in os.walk(input_dir)
                    for filename in filenames if filename.endswith('.hdr')]
        return filelist

if __name__ == '__main__':
