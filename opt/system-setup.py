#!/usr/bin/env python3

import sys
import os
import argparse

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "readies"))
import paella

#----------------------------------------------------------------------------------------------

class RedisAISetup(paella.Setup):
    def __init__(self, nop=False):
        paella.Setup.__init__(self, nop)

    def common_first(self):
        self.install_downloaders()
        self.setup_pip()
        self.pip_install("wheel virtualenv")
        self.pip_install("setuptools --upgrade")

        if self.os == 'linux':
            self.install("ca-certificates")
        self.install("git unzip wget patchelf")
        self.install("coreutils") # for realpath

    def debian_compat(self):
        self.pip_install("-IU --force-reinstall setuptools")
        self.install("build-essential cmake")
        self.install("python3-regex")
        self.install("python3-venv python3-psutil python3-networkx python3-numpy") # python3-skimage
        self.install_git_lfs_on_linux()

    def redhat_compat(self):
        self.install("redhat-lsb-core")
        self.run("%s/readies/bin/enable-utf8" % HERE)
        
        self.group_install("'Development Tools'")
        self.install("cmake3")
        self.run("ln -s `command -v cmake3` /usr/local/bin/cmake")
        
        self.install("centos-release-scl")
        self.install("devtoolset-8")
        self.run("cp /opt/rh/devtoolset-8/enable /etc/profile.d/scl-devtoolset-8.sh")
        paella.mkdir_p("%s/profile.d" % ROOT)
        self.run("cp /opt/rh/devtoolset-8/enable %s/profile.d/scl-devtoolset-8.sh" % ROOT)

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
        self.install("cmake")
        self.run("ln -s `command -v cmake3` /usr/local/bin/cmake")
        self.install("python3-venv python3-psutil python3-networkx")
        self.install_git_lfs_on_linux()

    def macosx(self):
        if sh('xcode-select -p') == '':
            fatal("Xcode tools are not installed. Please run xcode-select --install.")

        self.install_gnu_utils()
        self.install("cmake")
        self.install("git-lfs")
        self.install("redis")

    def common_last(self):
        self.run("python3 -m pip uninstall -y ramp-packer RLTest || true")
        # redis-py-cluster should be installed from git due to redis-py dependency
        self.pip_install("--no-cache-dir git+https://github.com/Grokzen/redis-py-cluster.git@master")
        self.pip_install("--no-cache-dir git+https://github.com/RedisLabsModules/RLTest.git@master")
        self.pip_install("--no-cache-dir git+https://github.com/RedisLabs/RAMP@master")

        self.pip_install("-r %s/readies/paella/requirements.txt" % HERE)
        self.pip_install("-r %s/test/test_requirements.txt" % ROOT)

        self.pip_install("awscli")
        self.pip_install("mkdocs mkdocs-material mkdocs-extensions")

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

RedisAISetup(nop=args.nop).setup()
