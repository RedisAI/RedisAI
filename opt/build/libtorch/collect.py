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

PYTORCH_VERSION = '1.2.0'

parser = argparse.ArgumentParser(description='Prepare RedisAI dependant distribution packages.')
parser.add_argument('--pytorch', default='pytorch', help='root of pytorch repository')
parser.add_argument('--pytorch-ver', default=PYTORCH_VERSION, help='pytorch version')
parser.add_argument('--dest', default='dest', help='destination directory')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

#----------------------------------------------------------------------------------------------

pytorch = Path(args.pytorch).resolve()
dest = Path(args.dest).resolve()

#----------------------------------------------------------------------------------------------

pt_build='cpu'

platform = paella.Platform()

pt_os = platform.os
if pt_os == 'macosx':
    pt_os = 'darwin'

pt_arch = platform.arch
if pt_arch == 'x64':
    pt_arch = 'x86_64'
elif pt_arch == 'arm64v8':
    pt_arch = 'arm64'
elif pt_arch == 'arm32v7':
    pt_arch = 'arm'

pt_ver = args.pytorch_ver

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

def collect_pytorch():
    d_pytorch = dest/'libtorch'
    with cwd(pytorch/'torch/include'):
        for f in Path('.').glob('**/*.h'):
            copy_p(f, d_pytorch/'include')
    with cwd(pytorch/'torch/lib'):
        for f in Path('.').glob('*.a'):
            copy_p(f, d_pytorch/'lib')
        for f in Path('.').glob('*.so*'):
            copy_p(f, d_pytorch/'lib')
    with cwd(pytorch/'torch'):
        shutil.copytree('share', d_pytorch/'share', ignore_dangling_symlinks=True)
    create_tar('libtorch-{}-{}-{}-{}.tar.gz'.format(pt_build, pt_os, pt_arch, pt_ver), dest, 'libtorch')

#----------------------------------------------------------------------------------------------

collect_pytorch()
