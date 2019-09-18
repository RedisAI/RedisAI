#!/usr/bin/env python3

import sys
import os
from subprocess import Popen, PIPE
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "readies"))
import paella

#----------------------------------------------------------------------------------------------

class RedisAISetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        self.install_downloaders()
        self.setup_pip()
        self.pip3_install("wheel")
        self.pip3_install("setuptools --upgrade")
        
        if self.os == 'linux':
            self.install("ca-certificates")
        self.install("git cmake unzip wget patchelf awscli")
        self.install("coreutils") # for realpath

    def debian_compat(self):
        self.install("build-essential")
        self.install("python3-venv python3-psutil python3-networkx python3-numpy python3-skimage")
        self.install_git_lfs_on_linux()

    def redhat_compat(self):
        self.group_install("'Development Tools'")
        self.install("redhat-lsb-core")

        self.install("epel-release")
        self.install("python36 python36-pip")
        self.install("python36-psutil")

        self.install_git_lfs_on_linux()
        
    def fedora(self):
        self.group_install("'Development Tools'")
        self.install("python3-venv python3-psutil python3-networkx python3-numpy")
        self.install_git_lfs_on_linux()

    def macosx(self):
        p = Popen('xcode-select -p', stdout=PIPE, close_fds=True, shell=True)
        out, _ = p.communicate()
        if out.splitlines() == []:
            fatal("Xcode tools are not installed. Please run xcode-select --install.")
        self.install("git-lfs")
        self.install("redis")

    def install_git_lfs_on_linux(self):
        self.run("curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash")
        self.install("git-lfs")
    
    def common_last(self):
        if not self.has_command("RLTest"):
            self.pip3_install("git+https://github.com/RedisLabsModules/RLTest.git@master")
        if not self.has_command("ramp"):
            self.pip3_install("git+https://github.com/RedisLabs/RAMP@master")
        root = os.path.join(os.path.dirname(__file__), "..")
        self.pip3_install("-r {}/test/test_requirements.txt".format(root))
        # self.pip3_install("redis-py-cluster")

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

RedisAISetup(nop = args.nop).setup()
