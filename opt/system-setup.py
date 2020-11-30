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
        # if self.osnick == 'xenial': 
        #     self.pip_install("setuptools --upgrade")
        #     self.pip_install("-IU --force-reinstall setuptools")

        if self.os == 'linux':
            self.install("ca-certificates")
        self.install("git unzip wget patchelf")
        self.install("coreutils") # for realpath

    def debian_compat(self):
        self.install("gawk")
        self.install("build-essential")
        self.install("libssl-dev")
        self.run("""
        apt remove -y --purge --auto-remove cmake
        version=3.19
        build=0
        mkdir ~/temp
        cd ~/temp
        wget https://cmake.org/files/v$version/cmake-$version.$build-Linux-x86_64.sh 
        mkdir /opt/cmake
        sh cmake-$version.$build-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
        ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
        """)
        self.install("clang-format")
        self.install("python3-regex")
        self.install("python3-psutil python3-networkx python3-numpy") # python3-skimage
        self.install_git_lfs_on_linux()

    def redhat_compat(self):
        self.install("redhat-lsb-core")
        self.run("%s/readies/bin/enable-utf8" % HERE)

        self.group_install("'Development Tools'")

        self.install("centos-release-scl")
        self.install("devtoolset-8")
        self.run("cp /opt/rh/devtoolset-8/enable /etc/profile.d/scl-devtoolset-8.sh")

        self.install("llvm-toolset-7")

        paella.mkdir_p("%s/profile.d" % ROOT)
        self.run("cp /opt/rh/devtoolset-8/enable %s/profile.d/scl-devtoolset-8.sh" % ROOT)

        self.run("""
            dir=$(mktemp -d /tmp/tar.XXXXXX)
            (cd $dir; wget -q -O tar.tgz http://redismodules.s3.amazonaws.com/gnu/gnu-tar-1.32-x64-centos7.tgz; tar -xzf tar.tgz -C /; )
            rm -rf $dir
            """)


        if not self.dist == "amzn":
            self.install("epel-release")
            self.install("python3-devel libaec-devel")
            self.install("python36-psutil")
        else:
            self.run("amazon-linux-extras install epel", output_on_error=True)
            self.install("python3-devel")
            self.pip_install("psutil")

        self.install("cmake3")
        self.run("ln -sf `command -v cmake3` /usr/local/bin/cmake")

        self.install_git_lfs_on_linux()

    def fedora(self):
        self.group_install("'Development Tools'")
        self.install("cmake")
        self.run("ln -sf `command -v cmake3` /usr/local/bin/cmake")
        self.install("python3 python3-psutil python3-networkx")
        self.install("clang")
        self.install_git_lfs_on_linux()

    def macos(self):
        if sh('xcode-select -p') == '':
            fatal("Xcode tools are not installed. Please run xcode-select --install.")

        self.install_gnu_utils()
        self.install("cmake")
        self.install("git-lfs")
        self.install("redis")
        self.install("clang-format")

    def common_last(self):
        self.run("python3 -m pip uninstall -y ramp-packer RLTest || true")
        # redis-py-cluster should be installed from git due to redis-py dependency
        self.pip_install("--no-cache-dir git+https://github.com/Grokzen/redis-py-cluster.git@master")
        self.pip_install("--no-cache-dir git+https://github.com/RedisLabsModules/RLTest.git@master")
        self.pip_install("--no-cache-dir git+https://github.com/RedisLabs/RAMP@master")

        self.pip_install("-r %s/readies/paella/requirements.txt" % HERE)
        self.pip_install("-r %s/tests/flow/test_requirements.txt" % ROOT)

        self.pip_install("awscli")
        self.pip_install("mkdocs mkdocs-material mkdocs-extensions")

#----------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Set up system for build.')
parser.add_argument('-n', '--nop', action="store_true", help='no operation')
args = parser.parse_args()

RedisAISetup(nop=args.nop).setup()
