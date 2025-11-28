#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='kitti_converter',
    version='1.0.0',
    description='Convert KITTI dataset to ROS2 bag file the easy way!',
    author='Tomas Krejci',
    author_email='tomas@krej.ci',
    url='https://github.com/aidanjones3/kitti_converter/',
    keywords = ['dataset', 'ros2', 'rosbag2', 'kitti'],
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['kitti_converter=kitti_converter.__main__:main'],
    },
    install_requires=['pykitti', 'progressbar2'],
    zip_safe=False,
)