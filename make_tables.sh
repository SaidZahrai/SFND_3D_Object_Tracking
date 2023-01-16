#!/usr/bin/bash
#

## TTC: HARRIS BRIEF #0012 TTC_Lidar: -10.854 lidarpoints: 300 TTC_Camera:  11.792 keypoints: 46

grep TTC $1 | awk '
{
    detectors[$2]
    descriptors[$3]
    images[$4]
    ttc_lidar[$2][$3][$4] = $6
    lidar_points[$2][$3][$4] = $8
    ttc_camera[$2][$3][$4] = $10
    keypoints[$2][$3][$4] = $12
}
END {
    printf "Detector,Extractor,Image,TTC_Lidar,Lidar_Points,TTC_Camera,Keypoints\n"
    for (det in detectors) {
        for (des in descriptors){
            for (im in images)
            {
                if (ttc_lidar[det][des][im])
                {
                    printf det
                    printf ","
                    printf des
                    printf ","
                    printf substr(im, 2, 4)
                    printf ","
                    printf ttc_lidar[det][des][im]
                    printf ","
                    printf lidar_points[det][des][im]
                    printf ","
                    printf ttc_camera[det][des][im]
                    printf ","
                    printf keypoints[det][des][im]
                    printf "\n"
                }
            }
        }
    }
}'
