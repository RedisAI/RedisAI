#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import shutil
import tarfile

# this refers to deps directory inside a container
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "readies"))
import paella

#----------------------------------------------------------------------------------------------

TFLITE_VERSION = '2.0.0'

parser = argparse.ArgumentParser(description='Prepare RedisAI dependant distribution packages.')
parser.add_argument('--tensorflow', default='tensorflow', help='root of tensorflow repository')
parser.add_argument('--version', default=TFLITE_VERSION, help='tensorflow version')
parser.add_argument('--dest', default='dest', help='destination directory')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

#----------------------------------------------------------------------------------------------

tensorflow = Path(args.tensorflow).resolve()
dest = Path(args.dest).resolve()

#----------------------------------------------------------------------------------------------

platform = paella.Platform()

tf_os = platform.os
tf_os_internal = tf_os
if tf_os == 'macos':
    tf_os = 'darwin'
    tf_os_internal = 'osx'

tf_arch = platform.arch
if tf_arch == 'x64':
    tf_arch = 'x86_64'
elif tf_arch == 'arm64v8':
    tf_arch = 'arm64'

tf_ver = args.version

#----------------------------------------------------------------------------------------------

def copy_p(src, dest):
    f = dest/src
    paella.mkdir_p(os.path.dirname(f))
    shutil.copy(src, f, follow_symlinks=False)

def create_tar(name, basedir, dir='.'):
    def reset_uid(tarinfo):
        tarinfo.uid = tarinfo.gid = 0
        tarinfo.uname = tarinfo.gname = "root"
        return tarinfo
    with cwd(basedir):
        with tarfile.open(name, 'w:gz') as tar:
            tar.add(dir, filter=reset_uid)

def collect_tflite():
    d_tensorflow = dest
    with cwd(tensorflow):
        for f in Path('tensorflow/lite').glob('**/*.h'):
            copy_p(f, d_tensorflow/'include')
        with cwd('tensorflow/lite/tools/make'):
            with cwd('downloads/flatbuffers/include'):
                for f in Path('.').glob('**/*.h'):
                    copy_p(f, d_tensorflow/'include')
            with cwd(f'gen/{tf_os_internal}_{tf_arch}/lib'):
                for f in Path('.').glob('*.a'):
                    copy_p(f, d_tensorflow/'lib')
    create_tar(dest/f'libtensorflowlite-{tf_os}-{tf_arch}-{tf_ver}.tar.gz', dest)

#----------------------------------------------------------------------------------------------

collect_tflite()
