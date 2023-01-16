
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

/********************************************************************************************/

    //// Default settings:
    bool bVis = false;            // visualize results
    string detectorType = "FAST"; // "FAST"; // "ORB"; // "HARRIS"; // "SIFT"; // "SHITOMASI"; // "AKAZE"; // 
    string descriptorType = "BRIEF"; // "BRIEF"; // "BRISK"; // "FREAK"; // "ORB"; //  "AKAZE"; // "SIFT"; //
    string matcherType = "MAT_BF";        // "MAT_BF"; // "MAT_FLANN"; //
    string selectorType = "SEL_KNN";       // "SEL_NN"; // "SEL_KNN"; //

    if (argc == 1)
    {
        cout << "You have started the program without any arguments. This is similar to:\n";
        cout << "\t3D_object_tracking FAST BRIEF MAT_BF SEL_KNN OFF\n";
        cout << "meaning you intend to run 2D_feature_tracking with\n";
        cout << "\tDetector: FAST [HARRIS/SHITOMASI/FAST/ORB/AKAZE/SIFT]\n";
        cout << "\tDescriptor: BRIEF [BRIEF/BRISK/FREAK/ORB/AKAZE/SIFT]\n";
        cout << "\tMatcher: MAT_BF [MAT_BF/MAT_FLANN]\n";
        cout << "\tSelector: SEL_KNN [SEL_NN/SEL_KNN]\n";
        cout << "\tVisualization: OFF [ON/OFF]\n";
        cout << "\n";
        cout << "Please try with other combinations of interest!\n";
    } 
    else if (argc == 6)
    {
        detectorType = string(argv[1]);
        if (!((detectorType.compare("HARRIS") == 0) || (detectorType.compare("SHITOMASI")) || (detectorType.compare("FAST")) ||
            (detectorType.compare("ORB") == 0) || (detectorType.compare("AKAZE") == 0) || (detectorType.compare("SIFT") == 0)))
        {
            cout << detectorType << " not recognized. Please check your choice for detector.\n";
            cout << "\tDetector: [HARRIS/SHITOMASI/FAST/ORB/AKAZE/SIFT]\n";
            return 1;
        }
        descriptorType = string(argv[2]);
        if (!((descriptorType.compare("BRIEF") == 0) || (descriptorType.compare("BRISK")) || (descriptorType.compare("FREAK")) ||
            (descriptorType.compare("ORB") == 0) || (descriptorType.compare("AKAZE") == 0) || (descriptorType.compare("SIFT") == 0)))
        {
            cout << descriptorType << " not recognized. Please check your choice for descriptor.\n";
            cout << "\tDescriptor: [BRIEF/BRISK/FREAK/ORB/AKAZE/SIFT]\n";
            return 1;
        }
        matcherType = string(argv[3]);
        if (!((matcherType.compare("MAT_BF") == 0) || (matcherType.compare("MAT_FLANN") == 0)))
        {
            cout << matcherType << " not recognized. Please check your choice for matcher.\n";
            cout << "\tMAT_BF [MAT_BF/MAT_FLANN]\n";
            return 1;
        }
        selectorType = string(argv[4]);
        if (!((selectorType.compare("SEL_NN") == 0) || (selectorType.compare("SEL_KNN") == 0)))
        {
            cout << selectorType << " not recognized. Please check your choice for selector.\n";
            cout << "\tSelector: [SEL_NN/SEL_KNN]\n";
            return 1;
        }
        bVis = string(argv[5]).compare("ON") == 0;
        if (!((string(argv[5]).compare("ON") == 0) || (string(argv[5]).compare("OFF") == 0)))
        {
            cout << string(argv[5]) << " not recognized. Please check your choice for visualization.\n";
            cout << "\tVisualization: [ON/OFF]\n";
            return 1;
        }
    }
    else
    {
        cout << "You have entered " << argc << " arguments:" << "\n";
    
        for (int i = 0; i < argc; ++i)
            cout << argv[i] << "\n";

        cout << "Please execute\n";
        cout << "\t2D_feature_tracking Detector Descriptor Matcher Selector Visualization\n";
        cout << "with one of the following choices:\n";
        cout << "\tDetector: [HARRIS/SHITOMASI/FAST/ORB/AKAZE/SIFT]\n";
        cout << "\tDescriptor: [BRIEF/BRISK/FREAK/ORB/AKAZE/SIFT]\n";
        cout << "\tMatcher: [MAT_BF/MAT_FLANN]\n";
        cout << "\tSelector: [SEL_NN/SEL_KNN]\n";
        cout << "\tVisualization: [ON/OFF]\n";
        return 1;
    }

    //// Consistency check:
    if ((descriptorType.compare("AKAZE") == 0) && (detectorType.compare("AKAZE") != 0))
    {
        std::cout << "AKAZE descriptor requires AKAZE detector. Execution aborted.\n";
        return 1;
    }

    cout << "*** Start of execution of the command  ";
    for (int i = 0; i < argc; ++i)
        cout << argv[i] << " " ;
    cout <<  " with OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << "\n";
/********************************************************************************************/

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        if (bVis)
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        // bVis = false;
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);
        // bVis = false;

        if (bVis)
            cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        if (bVis)
            cout << "#3 : CROP LIDAR POINTS done" << endl;

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        // bVis = true;
        if(bVis)
        {
            bool bWait = false;
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(1000, 1000), bWait);
        }
        // bVis = false;

        if (bVis)
            cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

        /* DETECT IMAGE KEYPOINTS */
        ////
        //// This part is taken from mid-term project with minor changes for adaptation.
        ////
        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        double t;
        if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, t, false);
        }
        else if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, t, false);
        }
        else 
        {
            detKeypointsModern(keypoints, imgGray, detectorType, t, false);
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        if (bVis)
            cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, t);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        if (bVis)
            cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            // All descriptors have binary output vectors except SIFT which has a histogram based description vecctor
            string descriptorOutputType;
            if (descriptorType.compare("SIFT") == 0)
            {
                descriptorOutputType =  "DES_HOG";
            }
            else 
            {
                descriptorOutputType =  "DES_BINARY";
            }

            vector<cv::DMatch> matches;
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorOutputType, matcherType, selectorType, t);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            if (bVis)
                cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;
            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> list of 3D objects (vector<BoundingBox>) matched between current and previous frame (matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            if (bVis)
                cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" <<  endl;

            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check whether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> time-to-collision is computed based on Lidar data (computeTTCLidar)
                    double ttcLidar; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// TASK FP.3 -> Enclosed keypoint matches assigned to bounding box (clusterKptMatchesWithROI)
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);   
                                    
                    //// TASK FP.4 -> time-to-collision based on camera (computeTTCCamera)
                    double ttcCamera;
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    // bVis = true;
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        int textScale = 2;
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80/textScale, 50), cv::FONT_HERSHEY_PLAIN, 2/textScale, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    // bVis = false;

                     printf ("TTC: %s %s #%s TTC_Lidar: %7.3f lidarpoints: %d TTC_Camera: %7.3f keypoints: %d\n", detectorType.c_str(), descriptorType.c_str(), 
                            imgNumber.str().c_str(), ttcLidar, (int) currBB->lidarPoints.size(), ttcCamera, (int) currBB->kptMatches.size());
                } // eof TTC computation
            } // eof loop over all BB matches            

        }

    } // eof loop over all images


    cout << "***  End of execution of the command  ";
    for (int i = 0; i < argc; ++i)
        cout << argv[i] << " " ;
    cout <<  "\n";
    cout << "***" << endl;

    return 0;
}
