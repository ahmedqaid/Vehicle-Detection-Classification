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
const string pathToVid = "C:\\Users\\q041\\OneDrive\\Desktop\\ippr-vid1.mp4";

Mat KMeans(Mat original, int clusters);
vector<Mat> applySegmentation(Mat processed, Mat original);
void classify(Mat object);

int main()
{

		VideoCapture cap;
		cap.open(pathToVid);
	
		if (!cap.isOpened()) {
			cout << "Problem with opening video!" << endl;
			return -1;
		}
	
		//Mat frame;
		Mat salient;
		Mat kMeans;
		Mat denoised;
	
		int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
		int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		VideoWriter video("C:\\Users\\salma\\Videos\\ProcessedVideo.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 10, Size(frame_width, frame_height));
	
		int frameCount = 0;
		//Ptr<StaticSaliencyFineGrained> SS = StaticSaliencyFineGrained::create();
	
		while (1) {
			Mat frame;
			Mat frame1;
			Mat diff;
			if (frameCount % 100 != 0 || frameCount < 4000) {
				frameCount++;
				video.write(frame);
				continue;
			}
			cap >> frame;
			cap >> frame1;
			cvtColor(frame, frame, COLOR_BGR2GRAY);
			cvtColor(frame1, frame1, COLOR_BGR2GRAY);
	
			if (frame.empty()) {
				break;
			}
			Ptr<StaticSaliencySpectralResidual> SS = StaticSaliencySpectralResidual::create();
			//Ptr<StaticSaliencyFineGrained> SS = StaticSaliencyFineGrained::create();
	



			SS->computeSaliency(frame, salient);
			salient.convertTo(salient, CV_8U, 255);

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
			vector<Mat> objects = applySegmentation(denoised, frame);
			for (Mat object : objects) {
				imshow("object", object);
				classify(object);
			}

			rectangle(frame, Point(10, 2), Point(100, 20), Scalar(255, 255, 255), -1);
			stringstream ss;
			ss << cap.get(CAP_PROP_POS_FRAMES);
			string frameNumberString = ss.str();
			//get the frame number and write it on the current frame
			putText(frame, frameNumberString.c_str(), Point(15, 15),
				FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			video.write(frame);/*writing the frame into the output video*/
			imshow("frame", frame);/*show the output frame*/
			waitKey(10);










			//Mat thresh;
			//Mat ret;
	
			//threshold(diff, thresh, 30, 255, THRESH_BINARY);
			//threshold(diff, ret, 30, 255, THRESH_BINARY);
	
			//imshow("t", thresh);
			//imshow("ret", ret);
	
	
	
	
			//waitKey(0);
			//return 0;
	
			//SS->computeSaliency(frame, salient);
			//salient.convertTo(salient, CV_8U, 255);
			//cvtColor(salient, salient, COLOR_GRAY2BGR);
			//kMeans = KMeans(salient, 3);
			//fastNlMeansDenoising(kMeans, denoised, 100, 7, 21); // was 35
	
			//dilate(denoised, denoised, getStructuringElement(MORPH_RECT, Size(1, 10 * 2 + 1), Point(0, 20))); // was 10
			//Mat dilate = denoised;
			//imshow("dilate", dilate);
			//erode(denoised, denoised, getStructuringElement(MORPH_RECT, Size(5 * 2 + 1, 1), Point(10, 0))); // was 5
	
			///////applySegmentation(thresh, frame);
			//imshow("a", denoised);
			//frameCount++;
			//video.write(frame);
	
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
	// convert back from BGR to grai color mode /
	cvtColor(output, output, COLOR_BGR2GRAY);
	return output;
	// return the output image to be used for the nejt section /
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
		/*if (rect.width < original.cols * 0.05
			|| rect.height < original.rows * 0.1
			|| rect.y < original.rows * 0.25
			|| rect.x < original.cols * 0.3
			|| wh > 3.0) {
			continue;
		}*/
		//if (rect.width < 300 || rect.width > rect.height * 2.5) {
		//	continue;
		//}
		if (rect.width < 300 || rect.width > rect.height * 2.5) {
			continue;
		}
		imshow(to_string(i), original(rect));
		cout << i << ": " << wh << " " << rect.width << " " << rect.height << " " << rect.x << " " << rect.y << endl;
		separateImages.push_back(original(rect));
	}
	return separateImages;
}

void classify(Mat object)
{
	/*read the file path*/
	string ModelFile = "bvlc_googlenet.caffemodel";
	String ConfigFile = "bvlc_googlenet.prototxt";
	String ClassifyFile = "classification_classes_ILSVRC2012.txt";
	/*read the networking layor*/
	Net net = readNet(ModelFile, ConfigFile);

	if (net.empty())
	{
		cout << "\nERORR: There is no layer in the network" << endl;
	}
	/*read the classify tex file*/
	fstream fs(ClassifyFile.c_str(), fstream::in);
	if (!fs.is_open())
	{
		cout << "\nERORR: Cannot load the class names\n";

	}
	/*create classes vector to store the lines*/
	vector<string> classes;
	string line;
	while (getline(fs, line))
	{
		classes.push_back(line);
	}
	fs.close();
	Mat blob = blobFromImage(object, 1, Size(224, 224), Scalar(104, 117, 123));
	if (blob.empty())
		cout << "\nERORR: Cannot create blob\n";
	net.setInput(blob);
	Mat prob = net.forward();
	// Determine the best four classes
	Mat sorted_idx;
	sortIdx(prob, sorted_idx, SORT_EVERY_ROW + SORT_DESCENDING);
	for (int i = 0; i < 4; ++i) {
		cout << classes[sorted_idx.at<int>(i)] << "\n - ";
		cout << "\n Probability: " << prob.at<float>(sorted_idx.at<int>(i)) << endl;
	}
	cout << "\nBest Probability: " << classes[sorted_idx.at<int>(0)] << "\n - ";
	//Draw a rectangle displaying the bounding box
	rectangle(object, Point(0, 100), Point(100, 10), Scalar(0, 0, 255));
	//draw rectangle above the vehicle
	rectangle(object, Point(0, 2), Point(200, 20), Scalar(255, 255, 255), -1);
	putText(object, classes[sorted_idx.at<int>(0)], cvPoint(0, 15), FONT_HERSHEY_SIMPLEX,
		0.5, cvScalar(0, 0, 255), 1);
}
