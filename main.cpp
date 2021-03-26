#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/dnn/dnn.hpp"
#include <string>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include "opencv2/bgsegm.hpp"
#include <opencv2/saliency/saliencySpecializedClasses.hpp>
#include <core\types_c.h>

using namespace std;
using namespace cv;
using namespace saliency;
using namespace std;
using namespace cv;
using namespace dnn;

const string path = "C:\\Users\\q041\\OneDrive\\Desktop\\Screenshot.png";
const string pathToVid = "C:\\Users\\q041\\OneDrive\\Desktop\\ippr-vid1.mp4";
vector<string> objs;
vector<Mat> applySegmentation(Mat processed, Mat original);
string classify(Mat object);

int frameCount = 0;
int carCount = 0;
int busCount = 0;
int truckCount = 0;
int bikeCount = 0;
int unknown = 0;

int main()
{
	VideoCapture cap;
	cap.open(pathToVid);

	if (!cap.isOpened()) {
		cout << "Problem with opening video!" << endl;
		return -1;
	}

	Mat denoised;

	int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
	int frames_per_second = cap.get(CAP_PROP_FPS);
	VideoWriter video("C:\\Users\\q041\\OneDrive\\Desktop\\ippr-vids.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), frames_per_second, Size(frame_width, frame_height));

	int frameCount = 0;

	while (1) {
		frameCount++;
		Mat frame;
		Mat theOGframe;
		Mat frame1;
		Mat diff;

		cap >> frame;
		cap >> frame1;
		theOGframe = frame;
		if (frame.empty() || frame1.empty()) {
			break;
		}

		cvtColor(frame, frame, COLOR_BGR2GRAY);
		cvtColor(frame1, frame1, COLOR_BGR2GRAY);

		Mat thresh;
		Mat ret;

		absdiff(frame1, frame, diff);
		medianBlur(frame, frame, 3);
		medianBlur(frame1, frame1, 3);
		threshold(diff, thresh, 30, 255, THRESH_BINARY);

		dilate(thresh, thresh, getStructuringElement(MORPH_CROSS, Size(1, 10 * 2 + 1), Point(0, 10)));
		erode(thresh, thresh, getStructuringElement(MORPH_CROSS, Size(5 * 2 + 1, 1), Point(5, 0)));

		imshow("dif", diff);
		imshow("thresh", thresh);
		
		vector<Mat> objects = applySegmentation(thresh, theOGframe);
		for (Mat object : objects) {
			//imshow("object", object);
			string classified = classify(object);
			if (classify(object).find("moving van") != std::string::npos) {
				truckCount++;
			}
			else if (classified.find("car") != std::string::npos
				|| classified.find("police") != std::string::npos
				|| classified.find("taxi") != std::string::npos
				|| classified.find("van") != std::string::npos
				|| classified.find("model") != std::string::npos) {
				carCount++;
			}
			else if (classified.find("bus") != std::string::npos) {
				busCount++;
			}
			else if (classified.find("truck") != std::string::npos) {
				truckCount++;
			}
			else if (classified.find("bike") != std::string::npos
				|| classified.find("tri") != std::string::npos
				|| classified.find("ricksh") != std::string::npos
				|| classified.find("helmet") != std::string::npos) {
				bikeCount++;
			}
			else {
				unknown++;
			}
		}

		rectangle(theOGframe, Point(10, 2), Point(100, 20), Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << cap.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(theOGframe, frameNumberString.c_str(), Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		video.write(theOGframe);
		imshow("frame", theOGframe);

		cout << "Car Count: " << carCount << endl;
		cout << "Truck Count: " << truckCount << endl;
		cout << "Bike Count: " << bikeCount << endl;
		cout << "Bus Count: " << busCount << endl;
		cout << "Unknown Count: " << unknown << endl;

		waitKey();
	}
	cap.release();
	video.release();
	destroyAllWindows();
	waitKey(0);
	return 0;
}

vector<Mat> applySegmentation(Mat processed, Mat original) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Mat> separateImages;
	threshold(processed, processed, 0, 255, THRESH_BINARY | THRESH_OTSU);
	int erosion = 5;
	erode(processed, processed, getStructuringElement(MORPH_RECT, Size(erosion * 2 + 1, 1), Point(erosion, 0)));
	int dilation = 10;
	dilate(processed, processed, getStructuringElement(MORPH_RECT, Size(1, dilation * 2 + 1), Point(0, dilation)));
	findContours(processed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	Mat drawing = Mat::zeros(processed.size(), CV_8UC3);
	RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++) {
		Scalar scalar = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		Rect rect = boundingRect(contours.at(i));
		double wh = rect.width / rect.height;
		if (rect.width < 300 || rect.width > rect.height * 2.5) {
			continue;
		}
		//imshow(to_string(i), original(rect));
		//cout << i << ": " << wh << " " << rect.width << " " << rect.height << " " << rect.x << " " << rect.y << endl;
		separateImages.push_back(original(rect));
	}
	return separateImages;
}

string classify(Mat object)
{
	String model = "bvlc_googlenet.caffemodel";
	String config = "bvlc_googlenet.prototxt";
	String classify = "classification_classes_ILSVRC2012.txt";

	Net net = readNet(model, config);
	if (net.empty()) {
		cout << "Error: Empty file!" << endl;
	}
	fstream fs(classify.c_str(), fstream::in);
	if (!fs.is_open()) {
		cout << "Error: Classification not available" << endl;
	}
	
	vector<string> classifiedObjects;
	string line;
	while (getline(fs, line)) {
		classifiedObjects.push_back(line);
	}
	fs.close();
	Mat blobFromImg = blobFromImage(object, 1, Size(224, 224), Scalar(104, 117, 123));
	if (blobFromImg.empty())
		cout << "Error: Blob is not available" << endl;
	net.setInput(blobFromImg);
	Mat probabilities = net.forward();

	Mat sorted_idx;
	sortIdx(probabilities, sorted_idx, SORT_EVERY_ROW + SORT_DESCENDING);
	for (int i = 0; i < 4; ++i) {
		//cout << classifiedObjects[sorted_idx.at<int>(i)] << endl;
		//cout << "\n Classified: " << probabilities.at<float>(sorted_idx.at<int>(i)) << endl;
	}
	//cout << "Best result: " << classifiedObjects[sorted_idx.at<int>(0)] << endl;
	//objs.push_back(classifiedObjects[sorted_idx.at<int>(0)]);

	rectangle(object, Point(0, 200), Point(200, 0), Scalar(0, 255, 0));
	rectangle(object, Point(0, 2), Point(200, 20), Scalar(255, 255, 255), -1);
	putText(object, classifiedObjects[sorted_idx.at<int>(0)], cvPoint(0, 15), FONT_HERSHEY_TRIPLEX, 0.5, cvScalar(0, 0, 0), 1);
	return classifiedObjects[sorted_idx.at<int>(0)];
}
