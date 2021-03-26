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
#include <core\types_c.h>

using namespace std;
using namespace cv;
using namespace saliency;
using namespace std;
using namespace cv;
using namespace dnn;

const string path = "C:\\Users\\q041\\OneDrive\\Desktop\\Screenshot.png";
const string pathToVid = "C:\\Users\\q041\\OneDrive\\Desktop\\ippr-vid2.mp4";

Mat KMeans(Mat original, int clusters);
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

	Mat salient;
	Mat kMeans;
	Mat denoised;

	int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
	int frames_per_second = cap.get(CAP_PROP_FPS);
	VideoWriter video("C:\\Users\\q041\\OneDrive\\Desktop\\ippr-vids.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), frames_per_second, Size(frame_width, frame_height));

	int frameCount = 0;

	while (1) {
		frameCount++;
		Mat frame;
		//if (frameCount % 100 != 0 || frameCount < 4000) {
		//	frameCount++;
		//	video.write(frame);
		//	continue;
		//}
		cap >> frame;
		Mat theOGImage = frame;
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		
		if (frame.empty()) {
			break;
		}
		Ptr<StaticSaliencySpectralResidual> SS = StaticSaliencySpectralResidual::create();
		//Ptr<StaticSaliencyFineGrained> SS = StaticSaliencyFineGrained::create();

		SS->computeSaliency(frame, salient);
		salient.convertTo(salient, CV_8U, 255);
		imshow("salient", salient);

		cvtColor(salient, salient, COLOR_GRAY2BGR);

		imshow("Original", frame);

		kMeans = KMeans(salient, 3);

		imshow("KMeans", kMeans);

		fastNlMeansDenoising(kMeans, denoised, 40, 7, 21);
		dilate(denoised, denoised, getStructuringElement(MORPH_RECT, Size(1, 10 * 2 + 1), Point(0, 10)));
		Mat dilate = denoised;
		imshow("dilate", dilate);
		erode(denoised, denoised, getStructuringElement(MORPH_RECT, Size(5 * 2 + 1, 1), Point(5, 0)));

		imshow("Denoised", denoised);
		vector<Mat> objects = applySegmentation(denoised, theOGImage);
		for (Mat object : objects) {
			imshow("object", object);
			classify(object);
		}

		rectangle(theOGImage, Point(10, 2), Point(100, 20), Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << cap.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(theOGImage, frameNumberString.c_str(), Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
		video.write(theOGImage);
		imshow("frame", theOGImage);
		waitKey();

		char c = (char)waitKey(1);
		if (c == 27)
			break;
		if (frameCount == 24000) {
			waitKey(0);
			break;

		}
	}
}


Mat KMeans(Mat original, int clusters) {
	int attempts;
	Mat labels;
	Mat centers;
	Mat samples(original.rows * original.cols, clusters, CV_32F);
	for (int i = 0; i < original.rows; i++)
	{
		for (int j = 0; j < original.cols; j++)
		{
			for (int k = 0; k < clusters; k++)
			{
				samples.at<float>(i + j * original.rows, k) = original.at<Vec3b>(i, j)[k];
			}
		}
	}

	attempts = 5;
	kmeans(samples, clusters, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 1.0), attempts, KMEANS_PP_CENTERS, centers);

	Mat output = Mat::zeros(original.size(), original.type());
	Vec2i pointVal = { 0, 0 };

	for (int i = 0; i < centers.rows; i++)
	{
		int sum = 0;
		for (int j = 0; j < centers.cols; j++)
		{
			sum += centers.at<float>(i, j);
		}
		if (sum / 3 > pointVal[1]) {
			pointVal[0] = i;
			pointVal[1] = sum / 3;
		}
	}

	for (int i = 0; i < original.rows; i++)
		for (int j = 0; j < original.cols; j++)
		{
			int cluster_idj = labels.at<int>(i + j * original.rows, 0);
			if (cluster_idj == pointVal[0]) {
				output.at<Vec3b>(i, j)[0] = centers.at<float>(cluster_idj, 0);
				output.at<Vec3b>(i, j)[1] = centers.at<float>(cluster_idj, 1);
				output.at<Vec3b>(i, j)[2] = centers.at<float>(cluster_idj, 2);
			}
		}
	cvtColor(output, output, COLOR_BGR2GRAY);
	return output;
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
		imshow(to_string(i), original(rect));
		cout << i << ": " << wh << " " << rect.width << " " << rect.height << " " << rect.x << " " << rect.y << endl;
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
