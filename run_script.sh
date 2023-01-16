#! /usr/bin/bash

echo "Looping over all detectors, descriptor extractors, matchers and selectors"

for matcher in MAT_BF 
do
    for selector in SEL_KNN  
    do
        for extractor in BRIEF BRISK FREAK SIFT  
        do
            for detector in HARRIS SHITOMASI FAST ORB AKAZE SIFT  
            do
                ./3D_object_tracking $detector $extractor $matcher $selector OFF
            done
        done
        for detector in HARRIS SHITOMASI FAST ORB AKAZE  
        do
            ./3D_object_tracking $detector ORB $matcher $selector OFF
        done
        ./3D_object_tracking AKAZE AKAZE $matcher $selector OFF
    done
done