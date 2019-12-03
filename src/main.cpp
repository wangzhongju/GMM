// This is based on the "An Improved Adaptive Background Mixture Model for
// Real-time Tracking with Shadow Detection" by P. KaewTraKulPong and R. Bowden
// Author : wangzhongju
// Date   : 2019-07-25
// Email  : wangzhongju@qq.com
 
#include "opencv2/opencv.hpp"
#include "MOG_BGS.h"
#include <iostream>
#include <cstdio>
#include <csignal>
#include <sl/Camera.hpp>
 
#define USE_ZED ""
using namespace cv;
using namespace std;

bool key = false;

void signalHandler(int signum) {
    key = true;
    cout << "Interrupt signal (" << signum << ") received. \n";
}


int main(int argc, char* argv[])
{
	sl::Camera zed;
    sl::InitParameters mZedParams;
    signal(SIGINT, signalHandler);
    signal(SIGSEGV, signalHandler);
    signal(SIGTERM, signalHandler);

    //init zed params
    mZedParams.camera_fps = 30;
    mZedParams.camera_resolution = static_cast<sl::RESOLUTION>(3); //1:1080p 2:720p 3:VGA
    mZedParams.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD;
    mZedParams.coordinate_units = sl::UNIT_METER;
    mZedParams.depth_mode = static_cast<sl::DEPTH_MODE>(1);
    mZedParams.sdk_verbose = true;
    mZedParams.sdk_gpu_id = 0;
    mZedParams.depth_stabilization = 1;
    mZedParams.camera_image_flip = false;

	#ifdef USE_ZED
    cout << "<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>\n";
    sl::ERROR_CODE err = zed.open(mZedParams);
    if (err != sl::SUCCESS) {
        cout << toString(err) << endl;
        zed.close();
        return -1; // Quit if an error occurred
    }
    #endif
    // Print camera information
    printf("ZED Model                 : %s\n", toString(zed.getCameraInformation().camera_model).c_str());
    printf("ZED Serial Number         : %d\n", zed.getCameraInformation().serial_number);
    printf("ZED Firmware              : %d\n", zed.getCameraInformation().firmware_version);
    printf("ZED Camera Resolution     : %dx%d\n", (int) zed.getResolution().width, (int) zed.getResolution().height);
    printf("ZED Camera FPS            : %d\n", (int) zed.getCameraFPS());
    
    int serialnum = zed.getCameraInformation().serial_number;
    std::string serialnum_str = "camera_" + std::to_string(serialnum);
    // Read ZED Camera params
    int Matwidth = zed.getResolution().width;
    int Matheight = zed.getResolution().height;
    sl::CalibrationParameters zedParam;
    zedParam = zed.getCameraInformation(sl::Resolution(Matwidth, Matheight)).calibration_parameters; 

    double left_k1 = zedParam.left_cam.disto[0];   // k1
    double left_k2 = zedParam.left_cam.disto[1];   // k2
    double left_k3 = zedParam.left_cam.disto[4];   // k3
    double left_p1 = zedParam.left_cam.disto[2];   // p1
    double left_p2 = zedParam.left_cam.disto[3];   // p2

    double left_fx = static_cast<double>(zedParam.left_cam.fx);
    double left_cx = static_cast<double>(zedParam.left_cam.cx);
    double left_fy = static_cast<double>(zedParam.left_cam.fy);
    double left_cy = static_cast<double>(zedParam.left_cam.cy);
        
    // Create a Mat 
    sl::Mat zed_image;
    sl::Mat point_cloud;
    // cv::Mat zedMat;
    time_t cur_time, pre_time;
    cur_time = pre_time = time(0);
    cout << "wait for 5s -------------------->\n";
    sleep(5);

	Mat frame, gray, mask;
	VideoCapture capture;
	// capture.open("/home/wangzhongju/workspace/wzj/img_algorithm/GMM/data/demo_new.avi");
	capture.open(0);
 
	// if (!capture.isOpened())
	// {
	// 	cout<<"No camera or video input!\n"<<endl;
	// 	return -1;
	// }
 
	MOG_BGS Mog_Bgs;
	int count = 0;
 
	while (!key)
	{
		#ifdef USE_ZED
		if (zed.grab() == sl::SUCCESS)
		#else
		if (!key)
		#endif
		{
			count++;
			// get left image
			#ifdef USE_ZED
			zed.retrieveImage(zed_image, sl::VIEW_LEFT);
			frame = Mat((int)zed_image.getHeight(), (int)zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM_CPU));
			#else
			capture >> frame;
			#endif
			if (frame.empty())
				break;
			cvtColor(frame, gray, CV_RGB2GRAY);
		
			if (count == 1)
			{
				Mog_Bgs.init(gray);
				Mog_Bgs.processFirstFrame(gray);
				cout<<" Using "<<TRAIN_FRAMES<<" frames to training GMM..."<<endl;
			}
			else if (count < TRAIN_FRAMES)
			{
				Mog_Bgs.trainGMM(gray);
			}
			else if (count == TRAIN_FRAMES)
			{
				Mog_Bgs.getFitNum(gray);
				cout<<" Training GMM complete!"<<endl;
			}
			else
			{
				Mog_Bgs.testGMM(gray);
				mask = Mog_Bgs.getMask();
				morphologyEx(mask, mask, MORPH_OPEN, Mat());  //高级形态学变换，MORPH_OPEN为开运算，可清除一些小亮点，放大局部低亮度的区域
				erode(mask, mask, Mat(7, 7, CV_8UC1), Point(-1, -1));  // You can use Mat(5, 5, CV_8UC1) here for less distortion 腐蚀算法，
				dilate(mask, mask, Mat(7, 7, CV_8UC1), Point(-1, -1));  //删除对象边界的某些元素，结果使二值图像见减小一圈；膨胀算法相反。
				imshow("mask", mask);
			}
	
			imshow("input", frame);	
	
			if ( cvWaitKey(10) == 'q' )
				break;
		}
	}
 
	return 0;
}
