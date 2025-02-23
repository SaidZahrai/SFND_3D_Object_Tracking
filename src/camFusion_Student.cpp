
#include <iostream>
#include <algorithm>
#include <numeric>
#include <list>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "clustering.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        int textScale = 2;
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f((left-250)/textScale, (bottom+50)/textScale), cv::FONT_ITALIC, 2/textScale, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f((left-250)/textScale, (bottom+125)/textScale), cv::FONT_ITALIC, 2/textScale, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    struct kptref
    {
        cv::KeyPoint kp;
        cv::DMatch m;
        double sqDistanceSum;
    };

    vector<kptref> inside_kps;
    
    int index = 0;
    for (auto match : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr.at(match.trainIdx).pt))
        {
            kptref ref;
            ref.kp = kptsCurr.at(match.trainIdx);
            ref.m = match;
            ref.sqDistanceSum = 0.0;
            inside_kps.push_back(ref);
        }
    }
    if (inside_kps.size() > 0)
    {
        for (auto it1 = inside_kps.begin(); it1 != inside_kps.end();  it1++)
        {
            it1->sqDistanceSum = 0;
            for (auto it2 = inside_kps.begin(); it2 != inside_kps.end();  it2++)
            { 
                it1->sqDistanceSum += (it1->kp.pt.x - it2->kp.pt.x)*(it1->kp.pt.x - it2->kp.pt.x) +
                                    (it1->kp.pt.y - it2->kp.pt.y)*(it1->kp.pt.y - it2->kp.pt.y);
            }
        }
        // Outliers are identified and extracted using the 1.5xIQR rule
        std::sort(inside_kps.begin(), inside_kps.end(), [](kptref a, kptref b){return a.sqDistanceSum < b.sqDistanceSum;});
        long medIndex = floor(inside_kps.size() / 2.0);
        double medDist = inside_kps.size() % 2 == 0 ? 
        (inside_kps[medIndex - 1].sqDistanceSum + inside_kps[medIndex].sqDistanceSum) / 2.0 : inside_kps[medIndex].sqDistanceSum;
        long Q1Index = floor(medIndex / 2.0);
        double Q1Val = medIndex % 2 == 0 ? 
        (inside_kps[Q1Index - 1].sqDistanceSum + inside_kps[Q1Index].sqDistanceSum) / 2.0 : inside_kps[Q1Index].sqDistanceSum;
        long Q3Index = inside_kps.size() - floor(medIndex / 2.0);
        double Q3Val = medIndex % 2 == 0 ? 
        (inside_kps[Q3Index - 1].sqDistanceSum + inside_kps[Q3Index].sqDistanceSum) / 2.0 : inside_kps[Q3Index].sqDistanceSum;
        double IQR = Q3Val - Q1Val;
        double x1IQR = Q1Val - 1.5 * IQR, x3IQR = Q3Val + 1.5 * IQR;

        for (auto it1 = inside_kps.begin(); it1 != inside_kps.end();  it1++) 
        {
            if ((it1->sqDistanceSum > x1IQR) && (it1->sqDistanceSum < x3IQR))
            {
                boundingBox.kptMatches.push_back(it1->m);
                boundingBox.keypoints.push_back(it1->kp);
            }
        }
        // cout << " Number of keypoints outliers according to the 1.5xIQR rule: " << inside_kps.size() - boundingBox.keypoints.size() << endl;
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
    // compute distance ratios between all matched keypoints
    // This implementation is taken from course material with the mean replaced with median
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // The median ratio value is used as input for calculation of TTC
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    if (abs(medDistRatio - 1) < 1e-3)
    {
        TTC = 1e3;
    }
    else
    {
        TTC = dT / (medDistRatio - 1);
    }
}

// Compute time-to-collision (TTC) based on Lidar points
// The calculation based on the closest point distance was fluctuating considerably.
// An implementation is proposed to cluster out areas with lower number of Lidar points.
// Then, the closest point in the remaining clusters is used. 
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double minXPrev = 1e9, minXCurr = 1e9;
    double dT = 1 / frameRate; // time between two measurements in seconds

    //// Use Euclidean Clustering for filtering out segments with fewwer number of points.
    EuclideanClustering<LidarPoint> ecPrev(lidarPointsPrev), ecCurr(lidarPointsCurr);
    float clusterTolerance = 0.2;
    int minSize = 50, maxSize = 1000;

    //// An unisotropic box is used to find points close to each other. The box is 5 times smaller in x-direction.
    auto prev_cluster_indices = ecPrev.euclideanCluster({clusterTolerance/5, clusterTolerance, clusterTolerance}, minSize, maxSize);
    auto curr_cluster_indices = ecCurr.euclideanCluster({clusterTolerance/5, clusterTolerance, clusterTolerance}, minSize, maxSize);

    // cout << "Segmentation. Prev: " << prev_cluster_indices.size() << " Curr: " << curr_cluster_indices.size() << endl;

    // find closest distance to Lidar points within ego lane
    // For each segment, the points are first sorted and then either smaller or median point can be selected.
    for (auto it1 = prev_cluster_indices.begin(); it1 != prev_cluster_indices.end(); it1++)
    {
        vector<LidarPoint> segPoints;
        for (auto it2 = it1->begin(); it2 != it1->end(); it2++)
        {
            segPoints.push_back(lidarPointsPrev[*it2]);
        }
        std::sort(segPoints.begin(), segPoints.end(), [](LidarPoint a, LidarPoint b) {return a.x < b.x;});
        long medIndex = floor(segPoints.size() / 2.0);
        double medDist = segPoints.size() % 2 == 0 ? (segPoints[medIndex - 1].x + segPoints[medIndex].x) / 2.0 : segPoints[medIndex].x; // compute median dist. ratio to remove outlier influence

        // Minimum distance point:
        minXPrev = minXPrev > segPoints[0].x ? segPoints[0].x : minXPrev;
        // Median distance point:
        // minXPrev = minXPrev > medDist ? medDist : minXPrev;
    }
    for (auto it1 = curr_cluster_indices.begin(); it1 != curr_cluster_indices.end(); it1++)
    {
        vector<LidarPoint> segPoints;
        for (auto it2 = it1->begin(); it2 != it1->end(); it2++)
        {
            segPoints.push_back(lidarPointsCurr[*it2]);
        }
        std::sort(segPoints.begin(), segPoints.end(), [](LidarPoint a, LidarPoint b) {return a.x < b.x;});
        long medIndex = floor(segPoints.size() / 2.0);
        double medDist = segPoints.size() % 2 == 0 ? (segPoints[medIndex - 1].x + segPoints[medIndex].x) / 2.0 : segPoints[medIndex].x; // compute median dist. ratio to remove outlier influence

        // Minimum distance point:
        minXCurr = minXCurr > segPoints[0].x ? segPoints[0].x : minXCurr;
        // Median distance point:
        // minXCurr = minXCurr > medDist ? medDist : minXCurr;
    }

    // compute TTC from both measurements
    if (abs(minXPrev - minXCurr) < 1e-3)
    {
        TTC = 1e3;
    }
    else
    {
        TTC = minXCurr * dT / (minXPrev - minXCurr);
    }
}

// Match bounding boxes based on the number matched keypoints between them
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Outer loop over all bounding boxes in the current image
    for (auto bboxCurr : currFrame.boundingBoxes)
    {  
        // For each bounding box, make a list of all the indices of bounding boxes in the previous image, with keypoints that are
        // matched with the current bounding box. The number of repeated index shows the number of matched keypoint for that 
        // specific bounding box.
        std::list<int> boxMatches;
        for (auto match : matches)
        {
            if (bboxCurr.roi.contains(currFrame.keypoints[match.trainIdx].pt))
            {
                for (auto bboxPrev : prevFrame.boundingBoxes)
                {
                    if (bboxPrev.roi.contains(prevFrame.keypoints[match.queryIdx].pt))
                    {
                        boxMatches.push_back(bboxPrev.boxID);
                    }
                }
            }
        }
        if (boxMatches.size() > 0)
        {
            // The best match is the one with maximum number of occurance in the list. To find that,
            // count number of occurance for each unique index value.
            auto uniqueIndices(boxMatches);
            uniqueIndices.sort();
            uniqueIndices.unique();
    
            int bestIndex = *uniqueIndices.begin();
            int maxOccurance = std::count(boxMatches.begin(), boxMatches.end(), bestIndex);
            int occurance = 0;
            if (uniqueIndices.size() > 1)
            {
                for (auto itr1 = uniqueIndices.begin(); itr1 != uniqueIndices.end(); itr1++)
                {
                    occurance = std::count(boxMatches.begin(), boxMatches.end(), *itr1);
                    if (occurance > maxOccurance)
                    {
                        maxOccurance = occurance;
                        bestIndex = *itr1;
                    }
                }

            }
            bbBestMatches.insert(std::make_pair(bestIndex, bboxCurr.boxID));
        }
    }
}
