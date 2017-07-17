
import os
import glob
import numpy as np
from pydub import AudioSegment


import logging
l = logging.getLogger("pydub.converter")
l.setLevel(logging.DEBUG)
l.addHandler(logging.StreamHandler())

def mkdir(path):
    try:
        os.makedirs(path)
        print("mkdirs : "+path)
    except FileExistsError:
        pass
    except Exception as e:
        raise e

def change_ext(path, ext):
    return os.path.splitext(path)[0] + ext


# Working Directores
root_dir = os.path.dirname(os.path.abspath(__file__))
mp3_parentdir = os.path.join(root_dir, "mp3")
wav_parentdir = os.path.join(root_dir,"wav")
mkdir(wav_parentdir)


os.chdir(mp3_parentdir)
for mp3_refpath in glob.glob("*/*.mp3"):
    # path
    mp3_basedir = os.path.dirname(mp3_refpath) # a,b,c,..
    mp3_abspath = os.path.join(mp3_parentdir, mp3_refpath)
    wav_abspath = os.path.join(wav_parentdir, change_ext(mp3_refpath,".wav"))
    # make dir
    mkdir(os.path.dirname(wav_abspath))

    if os.path.exists(wav_abspath):
        continue

    source = AudioSegment.from_file(mp3_abspath)
    source.export(wav_abspath, format='mp3')
