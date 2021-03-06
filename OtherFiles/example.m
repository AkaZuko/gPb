%% Compute globalPb and hierarchical segmentation for an example image.

addpath(fullfile(pwd,'lib'));

%% 1. compute globalPb on a BSDS image (5Gb of RAM required)
clear all; close all; clc;

imgFile = 'data/text_small.png';
outFile = 'data/small_test_gPb.mat';

gPb_orient = globalPb(imgFile, outFile);

%% 2. compute Hierarchical Regions

% for boundaries
ucm = contours2ucm(gPb_orient, 'imageSize');
imwrite(ucm,'data/101087_ucm.bmp');

% for regions 
ucm2 = contours2ucm(gPb_orient, 'doubleSize');
save('data/101087_ucm2.mat','ucm2');

%% 3. usage example
clear all;close all;clc;

%load double sized ucm
load('data/101087_ucm2.mat','ucm2');

% convert ucm to the size of the original image
ucm = ucm2(3:2:end, 3:2:end);

% get the boundaries of segmentation at scale k in range [0 1]
k = 0.4;
bdry = (ucm >= k);

% get superpixels at scale k without boundaries:
labels2 = bwlabel(ucm2 <= k);
labels = labels2(2:2:end, 2:2:end);

figure;imshow('data/text_small.png');
figure;imshow(ucm);
imwrite(ucm,'data/small_101_ucm.png');
figure;imshow(bdry);
imwrite(bdry,'data/small_102_ucm.png');
figure;imshow(labels,[]);colormap(jet);
imwrite(labels, prism,'data/text_small_regions.png');

%% 4. compute globalPb on a large image:

clear all; close all; clc;

imgFile = 'data/small3.jpg';
outFile = 'data/small_test_big_gPb.mat';

gPb_orient = globalPb_pieces(imgFile, outFile);
delete(outFile);
figure; imshow(max(gPb_orient,[],3)); colormap(jet);


%% 5. See also:
%
%   grouping/run_bsds500.m for reproducing our results on the BSDS500  
%
%   interactive/example_interactive.m for interactive segmentation
%
%   bench/test_benchs.m for an example on using the BSDS500 benchmarks

