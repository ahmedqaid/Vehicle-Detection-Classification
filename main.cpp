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
vector<Mat> applySegmentation(Mat processed, Mat original);

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

	kMeans = KMeans(salient, 3);

	imshow("KMeans", kMeans);
	fastNlMeansDenoising(kMeans, denoised, 35, 7, 21);

	dilate(denoised, denoised, getStructuringElement(MORPH_RECT, Size(1, 10 * 2 + 1), Point(0, 10)));
	Mat dilate = denoised;
	imshow("dilate", dilate);
	erode(denoised, denoised, getStructuringElement(MORPH_RECT, Size(5 * 2 + 1, 1), Point(5, 0)));

	imshow("Denoised", denoised);

	applySegmentation(denoised, original);

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



vector<Mat> applySegmentation(Mat processed, Mat original) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Mat> separateImages;
	threshold(processed, processed, 0, 255, THRESH_BINARY | THRESH_OTSU);
	int erosion = 5;
	erode(processed, processed, getStructuringElement(MORPH_RECT, Size(erosion*2+1, 1), Point(erosion, 0)));
	int dilation = 10;
	dilate(processed, processed, getStructuringElement(MORPH_RECT, Size(1, dilation * 2 + 1), Point(0, dilation)));
	findContours(processed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(processed.size(), CV_8UC3);
	RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++) {
		Scalar scalar = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		Rect rect = boundingRect(contours.at(i));
		double wh = rect.width / rect.height;
		imshow(to_string(i), original(rect));
		if (rect.width < original.cols * 0.05
			|| rect.height < original.rows * 0.1
			|| rect.y < original.rows * 0.25
			|| rect.x < original.cols * 0.3
			|| wh > 3.0) {
			continue;
		}
		separateImages.push_back(original(rect));
	}
	return separateImages;
}
