<img margin-top="50px" align="right" width="20%" src="http://www.apulus.com/wp-content/uploads/2014/11/OpenCV-Logo.png" alt="OpenCV Logo">
# Installation-Compilation instructions for OpenCV 3.1.0

## Using Anaconda (Recommend Setup for tutorial)
`$ conda install -c https://conda.binstar.org/menpo opencv3`
<p>This will provide you with all the modules necessary to do the tutorial. Unfortunately it could be possible that ffmpeg woudl not be working with opencv, this will mean reading and writtign videos will not work. A alternative is to compile OpenCV youself. Perhaps there are other solutions out there. But in that case you will only miss a very small part of the tutorial.</p>

# Steps to compile OpenCV 3.1.0 on Ubuntu (Python 2.7)

<p>Unfortunately OpenCV needs to be compiled from source which can get a bit messy. Here I list the steps that I followed
on Ubuntu 14.04.</p>
<p>We will compile the standard OpenCV source files PLUS the extra modules that contain the face recognition modules
needed for the tutorial</p>
<p>These steps will download the files in your home folder, compile files and delete them once installation is finished</p>

## Installation in Windows and OSX
<p>Here are some links to installation instructions. I haven't tested them neither in Windows or OSX.</p>
- Windows: [Link 1](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html), [Link 2](http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/windows_install/windows_install.html)
- OSX: [Link 1](http://www.learnopencv.com/install-opencv-3-on-yosemite-osx-10-10-x/), [Link 2](http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/)

# Compilation instructions for OpenCV 3.1.0 on Ubuntu (Python 2.7)
## Installing pre-requisite packages
`$ apt-get update`<br>
`$ apt-get -y upgrade`<br>
`$ apt-get install -y build-essential cmake git pkg-config libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev libgtk2.0-dev
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev gfortran python2.7-dev`<br>
`$ pip install numpy`

## Downloading source files
`$ cd ~; git clone https://github.com/Itseez/opencv_contrib.git`<br>
`$ cd opencv_contrib; git checkout 3.1.0`<br>
`$ cd ~; git clone https://github.com/Itseez/opencv.git`<br>
`$ cd opencv; git checkout 3.1.0; mkdir build; cd build`<br>

## Compiling source files
```bash
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
	-D BUILD_EXAMPLES=ON ..
```
`$ make -j4` (The number is amout of processor cores in your system)<br>
`$ sudo make install`<br>
`$ sudo ldconfig`

## Check compiled file
`$ cd /usr/local/lib/python2.7`<br>
Check inside `dist-packages` or `site-packages` that the `cv2.so` exists. The folder containing the file will be 
used if using a virtual enviroment. In my case `cv2.so` is inside `dist-packages`. In the following section change the path
to `site-packages` if necessary.

## In case you are using a virtual enviroment
We will make a soft link to the `cv2.so` file<br>
`$ ln -s /usr/local/lib/python2.7/dist-packages/cv2.so 
<PATH TO VIRTUAL ENVIROMENT>/lib/python2.7/site-packages/cv2.so`

## Test
Steps for testing outside or inside the virtualenv. If it works outside and not inside then check the creation of the soft link command,
most probably there is some mistake in the path.<br>
`$ python`<br>
`>>> import cv2` (if successful you have compiled OpenCV correctly congrats!)<br>
`>>> import cv2.face` (if successfull you also have the extra modules, you are fully set up!)

## Remove downloaded files (Optional)
If you plan to develop your own applications then you might want to keep them.<br>
`$ rm -rf ~/opencv ~/opencv_contrib`
