import os
import subprocess
import cv2
import pandas as pd
import numpy as np

from local_library.Frame import Frame


class ExtractLandmarks:
    def __init__(self, source_type, file, save_repertory, verbose=False):
        self.source_type = source_type
        self.file = file
        self.save_repertory = save_repertory
        self.stream = {}
        self.verbose = verbose

        if self.verbose:
            print("\n----- Prepare architecture for files -----")

        if os.path.isdir(self.save_repertory):
            if self.verbose:
                print("\tFiles exist... Cleaning directory content...")
            subprocess.call(['rm', '-rf', self.save_repertory + '/'])

        try:
            if self.verbose:
                print("\tCreate export directory")
            subprocess.call(["mkdir", '-p', self.save_repertory])
        except OSError as e:
            if self.verbose:
                print(e)
            pass
        try:
            if self.verbose:
                print("\tCreate export/imagesCollection directory")
            subprocess.call(["mkdir", self.save_repertory + "/imagesCollection"])
        except OSError as e:
            if self.verbose:
                print(e)
            pass
        try:
            if self.verbose:
                print("\tCreate export/stream directory")
            subprocess.call(["mkdir", self.save_repertory + "/stream"])
        except OSError as e:
            if self.verbose:
                print(e)
            pass

    def read_file(self, call):
        if self.source_type == "video":
            if self.verbose:
                print("\n----- Reading video and calculate landmarks -----")
            stream = cv2.VideoCapture(self.file)

            nframe = 0
            grabbed, frame = stream.read()
            while grabbed:
                call(self, Frame(nframe, frame))
                nframe = nframe + 1
                grabbed, frame = stream.read()

        else:
            if self.verbose:
                print("\n----- Reading image collection and calculate landmarks -----")

            # See link below
            # https://stackoverflow.com/questions/2212643/python-recursive-folder-read

            # If your current working directory may change during script execution, it's recommended to
            # immediately convert program arguments to an absolute path. Then the variable root below will
            # be an absolute path as well. Example:
            # walk_dir = os.path.abspath(walk_dir)

            # TODO vérifier qu'on n'ouvre que des .pgm
            nframe = 0
            for root, subdirs, files in os.walk(self.file):
                for filename in files:
                    # Dirty file extension check
                    if (".pgm" in filename) or (".jpg" in filename):
                        # file_path = os.path.join(root, filename)
                        # print('\t- file %s (full path: %s)' % (filename, file_path))
                        frame = cv2.imread(str(os.path.join(root, filename)))
                        call(self, Frame(nframe, frame))
                        nframe = nframe + 1

        # return video

    def generate_image(self, name, _img):
        cv2.imwrite(self.save_repertory + "/imagesCollection/" + name + ".jpeg", _img)

    def add_stream(self, stream, data):
        if not (stream in self.stream):
            self.stream[stream] = pd.DataFrame()

        self.stream[stream] = self.stream[stream].append(data, ignore_index=True)

    def save_stream(self, stream, opt):
        if self.verbose:
            print("\n----- Save landmarks into " + self.save_repertory + "stream/" + stream + ".csv" + " -----")

        tmp = pd.Series(np.repeat(opt, len(self.stream[stream])))
        self.stream[stream]['Target'] = tmp.values
        self.stream[stream].to_csv(self.save_repertory + "/stream/" + stream + ".csv")
