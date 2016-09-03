function [] = getregion(imgFile)
%% Compute globalPb and hierarchical segmentation for an example image.
addpath(genpath('./'));
addpath(fullfile(pwd,'lib'));

%% 1. compute globalPb on a BSDS image (5Gb of RAM required)
%clear all; close all; clc;

%imgFile = 'workspace/small_test.png';
outFile = 'workspace/small_test_gPb.mat';

gPb_orient = globalPb(imgFile, outFile);

%% 2. compute Hierarchical Regions

% for boundaries
ucm = contours2ucm(gPb_orient, 'imageSize');
imwrite(ucm,'workspace/101087_ucm.bmp');

% for regions 
ucm2 = contours2ucm(gPb_orient, 'doubleSize');
save('workspace/101087_ucm2.mat','ucm2');

%% 3. usage example
clear all;close all;clc;

%load double sized ucm
load('workspace/101087_ucm2.mat','ucm2');

% convert ucm to the size of the original image
ucm = ucm2(3:2:end, 3:2:end);

% get the boundaries of segmentation at scale k in range [0 1]
k = 0.5;
bdry = (ucm >= k);

% get superpixels at scale k without boundaries:
labels2 = bwlabel(ucm2 <= k);
labels = labels2(2:2:end, 2:2:end);

imwrite(ucm,'workspace/small_101_ucm.png');
imwrite(bdry,'workspace/small_102_ucm.png');
imwrite(labels, prism,'workspace/small_test_regions.png');
exit(0);
