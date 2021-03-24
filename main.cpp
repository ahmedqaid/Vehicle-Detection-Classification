#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/photo.hpp>

using namespace std;
using namespace cv;
using namespace saliency;

const string path = "C:\\Users\\q041\\OneDrive\\Desktop\\Screenshot.png";
const string pathToVid = "C:\\Users\\q041\\OneDrive\\Desktop\\ippr-vid.mp4";

Mat KMeans(Mat original, int clusters);
vector<Mat> segment(Mat src, Mat ori);

int main()
{
	Mat original = imread(path);
	Mat salient;
	Mat kMeans;
	Mat denoised;

	//Ptr<StaticSaliencyFineGrained> SS = StaticSaliencyFineGrained::create();
	Ptr<StaticSaliencySpectralResidual> SS = StaticSaliencySpectralResidual::create();
	SS->computeSaliency(original, salient);
	imshow("salient", salient);
	salient.convertTo(salient, CV_8U, 255);

	cvtColor(salient, salient, COLOR_GRAY2BGR);

	imshow("Original", original);

	/*
	Mat image_grayscale = salient.clone();
	cvtColor(image_grayscale, image_grayscale, COLOR_BGR2GRAY);
	threshold(image_grayscale, image_grayscale, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("Thresh", image_grayscale);
	*/

	kMeans = KMeans(salient, 5);

	imshow("KMeans", kMeans);
	fastNlMeansDenoising(kMeans, denoised, 35, 7, 21);

	dilate(denoised, denoised, getStructuringElement(MORPH_RECT, Size(1, 10 * 2 + 1), Point(0, 10)));
	Mat dilate = denoised;
	imshow("dilate", dilate);
	erode(denoised, denoised, getStructuringElement(MORPH_RECT, Size(5 * 2 + 1, 1), Point(5, 0)));

	imshow("Denoised", denoised);

	//segment(salient, original);

	waitKey(0);
	return 0;
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





vector<Mat> segment(Mat src, Mat ori) {
	vector<std::vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Mat> croppedImg; /*declare the croppedimg vector to store all the croppedimg*/
	/*get the threshold*/
	threshold(src, src, 0, 255, THRESH_BINARY | THRESH_OTSU);
	/*get the erosion*/
	int erosion_size = 5;
	erode(src, src, getStructuringElement(MORPH_RECT, Size(
		erosion_size * 2 + 1, 1), Point(erosion_size, 0)));
	//imshow("erosion", src);
	/*get the dilation*/
	int dilation_size = 10;
	dilate(src, src, getStructuringElement(MORPH_RECT, Size(1,
		dilation_size * 2 + 1), Point(0, dilation_size)));
	//imshow("dilation", src);

	findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	// Original image clone
	RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		Rect r = boundingRect(contours.at(i));
		double ratio = r.width / r.height;
		if (r.width < ori.cols * 0.05 || r.height < ori.rows * 0.1 ||
			r.y < ori.rows * 0.25 || (r.x < ori.cols * 0.3) || ratio > 3.0) {
			continue;
		}
		croppedImg.push_back(ori(r));
	}
	return croppedImg;
}
