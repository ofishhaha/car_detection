//
//  main.cpp
//  car
//
//  Created by ymy on 16/3/15.
//  Copyright © 2016年 ymy. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <cv.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );

//String car_cascade_name = "/Users/ymy/Desktop/SVM_HOG.xml";//california
//String car_cascade_name = "/Users/ymy/Desktop/UIUC/cas2.xml";//UIUC
  String car_cascade_name = "/Users/ymy/Documents/SJTU_bachelor/PRP/car_detection/code/car/cars.xml";
CascadeClassifier car_cascade;
string window_name = "Capture - car detection";
RNG rng(12345);

int main(int argc, const char * argv[]) {
    
    //VideoCapture cap("/Users/ymy/Documents/SJTU_bachelor/PRP/car_detection/traffic.avi");//汽车正面检测 for video processing
    Mat frame;
    
    frame = imread("/Users/ymy/Documents/SJTU_bachelor/PRP/car_detection/example.jpg");
    //if( !car_cascade.load( car_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    car_cascade.load( car_cascade_name );
    //-- 2. Read the video stream
    //detectAndDisplay( frame );
   //读取视频
    /*for(;;)
    {
        cap >> frame;
        if( !frame.empty() )
        { detectAndDisplay( frame ); }
        else
        { printf(" --(!) No captured frame -- Break!"); break; }
        int c = waitKey(10);
        if( (char)c == 'c' ) { break; }
    }*/
    waitKey(0);
    return 0;
}

void detectAndDisplay( Mat frame )
{
    std::vector<Rect> cars;
    Mat frame_gray;
    //VideoWriter out;
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect car

    car_cascade.detectMultiScale( frame_gray, cars, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    for( size_t i = 0; i < cars.size(); i++ )
    {
        if(cars[i].width<130 or cars[i].height<65) continue;
        Point center( cars[i].x + cars[i].width*0.5, cars[i].y + cars[i].height*0.5 );
        ellipse( frame, center, Size( cars[i].width*0.5, cars[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        
        Mat faceROI = frame_gray( cars[i] );
        
    }
    //-- Show what you got
    imshow( window_name, frame );
    imwrite("/Users/ymy/Documents/SJTU_bachelor/PRP/car_detection/result.jpg",frame);
    
}
