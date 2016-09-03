#!/bin/bash
python resize.py $1 250 250
matlab -nodisplay -nosplash -nodesktop -r "getregion '$1'"
python release.py workspace/small_test_regions.png $1
