from setuptools import find_packages, setup

package_name = 'my_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wego',
    maintainer_email='changmin@wego-robotics.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bomb = my_pkg.bomb:main',
            'lidar_sub = my_pkg.lidar_sub:main',
            'camera_sub = my_pkg.camera_sub:main',
            'img_compressor = my_pkg.img_compressor:main',
            'test = my_pkg.test:main',    
            'lane_tracking = my_pkg.lane_tracking:main',
            'lidar_test = my_pkg.lidar_test:main',
            ],
    },
)
