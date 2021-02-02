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

TENSORFLOW_VERSION = '1.15.0'
VARIANT='cpu'

parser = argparse.ArgumentParser(description='Prepare RedisAI dependant distribution packages.')
parser.add_argument('--root', default='tensorflow', help='root of tensorflow repository')
parser.add_argument('--version', default=TENSORFLOW_VERSION, help='tensorflow version')
parser.add_argument('--variant', default=VARIANT, help='build variant')
parser.add_argument('--dest', default='dest', help='destination directory')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

#----------------------------------------------------------------------------------------------

tensorflow = Path(args.root).resolve()
dest = Path(args.dest).resolve()

#----------------------------------------------------------------------------------------------

tf_build=args.variant

platform = paella.Platform()

tf_os = platform.os
if tf_os == 'macos':
    tf_os = 'darwin'

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

def collect_tensorflow():
    d_tensorflow = dest #/'tensorflow'
    with cwd(tensorflow):
        for f in Path('tensorflow/c').glob('**/*.h'):
            copy_p(f, d_tensorflow/'include')
    with cwd(tensorflow/'bazel-bin'/'tensorflow'):
        for f in Path('.').glob('*.so*'):
            if str(f).endswith(".params"):
                continue
            copy_p(f, d_tensorflow/'lib')
    create_tar(dest/f'libtensorflow-{tf_build}-{tf_os}-{tf_arch}-{tf_ver}.tar.gz', dest)

#----------------------------------------------------------------------------------------------

collect_tensorflow()
