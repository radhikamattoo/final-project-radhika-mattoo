# Computer Graphics Final Project

[![Build Status](https://travis-ci.com/NYUCG2017/assignment-4-radhikamattoo.svg?token=DKU6y6MTDpMMtsxTr53h&branch=master)](https://travis-ci.com/NYUCG2017/assignment-4-radhikamattoo)

Radhika Mattoo, rm3485@nyu.edu

# Overview

This project uses OpenGL to render meshes with texture and bump mapping for a more realistic scene. The meshes used were found for free on  [TurboSquid](http://www.turbosquid.com).



# Instructions

* `git clone --recursive https://github.com/radhikamattoo/texture-and-bump-mapping.git`
* `cd texture-and-bump-mapping`
* `mkdir build`
* `cd build`
* `cmake ../`
* `make && ./FinalProject_bin`

Because I am calculating face & vertex partial derivatives for the Pear, the program window will take ~ 15-20 seconds to open.
