#!/usr/bin/env python3

import sys
import os
import argparse

HERE = os.path.abspath(os.path.dirname(__file__))
READIES = os.path.join(HERE, "readies")
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, READIES)
import paella

#----------------------------------------------------------------------------------------------

class RedisAISetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        self.install_downloaders()
        self.pip_install("wheel")

        self.install("git unzip patchelf")
        self.install("coreutils") # for realpath

    def debian_compat(self):
        self.install("gawk")
        self.install("build-essential")
        self.install("libssl-dev")
        self.install("python3-regex")
        self.install("python3-psutil python3-networkx python3-numpy")
        if self.platform.is_arm():
            self.install("python3-dev") # python3-skimage
        self.install("libmpich-dev libopenblas-dev") # for libtorch
        self.install_git_lfs_on_linux()

    def redhat_compat(self):
        self.install("redhat-lsb-core")
        self.run("%s/bin/enable-utf8" % READIES)

        self.run("%s/bin/getgcc --modern" % READIES)
        # self.install("llvm-toolset-7")

        if self.arch == 'x64':
            self.install_linux_gnu_tar()

        if not self.dist == "amzn":
            self.install("epel-release")
            self.install("python3-devel libaec-devel")
            self.install("python36-psutil")
        else:
            self.run("amazon-linux-extras install epel", output_on_error=True)
            self.install("python3-devel")
            self.pip_install("psutil")

        self.install_git_lfs_on_linux()

    def fedora(self):
        self.group_install("'Development Tools'")
        self.install("python3 python3-psutil python3-networkx")
        self.install_git_lfs_on_linux()

    def linux_last(self):
        self.install("valgrind")

    def macos(self):
        self.install_gnu_utils()
        self.install("git-lfs")
        self.install("redis")

    def common_last(self):
        self.run("%s/bin/getclang --format" % READIES)
        if self.platform == "arm":
            self.run("%s/bin/getcmake" % READIES)
        else:
            self.run("%s/bin/getcmake --no-repo" % READIES)

        self.run("{PYTHON} {READIES}/bin/getrmpytools".format(PYTHON=self.python, READIES=READIES))

        self.pip_install("-r %s/tests/flow/test_requirements.txt" % ROOT)

        self.pip_install("awscli")
        self.pip_install("mkdocs mkdocs-material mkdocs-extensions")

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

RedisAISetup(nop=args.nop).setup()
