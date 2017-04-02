#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

float ratioTest = 0.70f;
int numKeyPoints = 1500;

void showMatValue(Mat &img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			printf("%d\t", img.at<char>(i, j));
		}
		printf("\n");
	}
}

void showMatDoubleValue(Mat &img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			printf("%f\t", img.at<double>(i, j));
		}
		printf("\n");
	}
}

void readCalibrationMatrix(Mat &K, const char *filename) {
	FILE *fp;
	fp = fopen(filename, "r");
	if (!fp)
		exit(1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			float temp;
			fscanf(fp, "%f", &temp);
			K.at<double>(i, j) = temp;
		}
	}
	fclose(fp);
}

void readDistortionCoefficients(Mat &D, const char *filename) {
	FILE *fp;
	fp = fopen(filename, "r");
	if (!fp)
		exit(1);

	for (int i = 0; i < 5; i++) {
		float temp;
		fscanf(fp, "%f ", &temp);
		D.at<double>(i, 0) = temp;
	}
}

int main() {
   char *debug = getenv("DEBUG");

	// Read the images as grayscale
	Mat left = imread("left.jpg", 0);
	Mat right = imread("right.jpg", 0);

	if (debug) {
		namedWindow("Left", WINDOW_AUTOSIZE);
		namedWindow("Right", WINDOW_AUTOSIZE);
		imshow("Left", left);
		imshow("Right", right);
	}

	// Detect the keypoints using the SURF detector
	if (debug)
		printf("task: Detect the keypoints using SURF detector\n");
	int minHessian = 400;
	vector<KeyPoint> left_keypoints, right_keypoints;
	OrbFeatureDetector detector(minHessian);

	detector.detect(left, left_keypoints);
	detector.detect(right, right_keypoints);

	if (debug) {
		Mat img_keypoints_left, img_keypoints_right;
		drawKeypoints(left, left_keypoints, img_keypoints_left, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		drawKeypoints(right, right_keypoints, img_keypoints_right, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		imshow("Left Keypoints", img_keypoints_left);
		imshow("Right keypoints", img_keypoints_right);
	}

	// Calculate descriptors (feature vectors)
	if (debug)
		printf("task: Calculate descriptors (feature vectors)\n");
	OrbDescriptorExtractor extractor;

	Mat left_descriptors, right_descriptors;

	extractor.compute(left, left_keypoints, left_descriptors);
	extractor.compute(right, right_keypoints, right_descriptors);

	// Convert the descriptors to CV_32F for Flann Matcher
	if (debug)
		printf("task: Convert the descriptor to CV_32F for Flann Matcher\n");
	if (left_descriptors.type() != CV_32F)
		left_descriptors.convertTo(left_descriptors, CV_32F);
	if (right_descriptors.type() != CV_32F)
		right_descriptors.convertTo(right_descriptors, CV_32F);

	// Matching descriptor vectors using FLANN matcher
	if (debug)
		printf("task: Matching descriptor vectors using FLANN matcher\n");
	FlannBasedMatcher matcher;
	vector<DMatch> matches;

	matcher.match(left_descriptors, right_descriptors, matches);

	double max_dist = 0, min_dist = 100;

	// Quick calculation of max and min distances between keypoints
	if (debug)
		printf("task: Quick calculation of max and min distances keypoints\n");
	for (int i = 0; i < left_descriptors.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	if (debug) {
		printf("Max Dist: %f\n", max_dist);
		printf("Min Dist: %f\n", min_dist);
	}

	// Get the good matches ie. that have only little error value
	if (debug)
		printf("task: Get the good matches ie. that have only little error value\n");
	vector<DMatch> good_matches;
	for (int i = 0; i < left_descriptors.rows; i++)
		if (matches[i].distance <= max(10*min_dist, 0.02))
			good_matches.push_back(matches[i]);

	if (debug) {
		Mat img_matches;
		drawMatches(left, left_keypoints, right, right_keypoints,
					good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow("Good matches", img_matches);
	}

	// Get the vector of points which have a corresponding spot in both
	if (debug)
		printf("task: Get the vector of points which have a corresponding spot in both\n");
	vector<Point2f> left_imp_points, right_imp_points;

	for (size_t i = 0; i < good_matches.size(); i++) {
		left_imp_points.push_back(left_keypoints[good_matches[i].queryIdx].pt);
		right_imp_points.push_back(right_keypoints[good_matches[i].trainIdx].pt);
	}

	// Finding The Fundamental Matrix
	if (debug)
		printf("task: Finding the Fundamental Matrix\n");
	Mat F = findFundamentalMat(Mat(left_imp_points), Mat(right_imp_points), CV_FM_RANSAC);

	if (debug)
		showMatValue(F);

	// Read the camera calibration matrix
	if (debug)
		printf("task: Read the camera calibration matrix\n");
	Mat K = Mat::eye(3, 3, CV_64F);
	readCalibrationMatrix(K, "camera_intrinsic_matrix.txt");

	if (debug)
		showMatDoubleValue(K);

	// Read the camera distortion coefficients
	if (debug)
		printf("task: Read the camera distortion coefficients\n");
	Mat D = Mat::zeros(5, 1, CV_64F);
	readDistortionCoefficients(D, "camera_distortion_matrix.txt");

	if (debug)
		showMatDoubleValue(D);

	// Find the Essential Matrix
	Mat Kt = Mat::zeros(3, 3, CV_64F);
	if (debug)
		printf("task: Find the Essential Matrix\n");

	transpose(K, Kt);
	if (debug) {
		printf("Transpose of Camera Calibration Matrix K is:\n");
		showMatDoubleValue(Kt);
	}

	Mat F_ = Mat::zeros(F.rows, F.cols, CV_64F);
	for (int i = 0; i < F.rows; i++) {
		for (int j = 0; j < F.cols; j++) {
			F_.at<double>(i, j) = F.at<char>(i, j);
		}
	}

	Mat E = Mat::zeros(3, 3, CV_64F);
	E = Kt*F_*K;

	if (debug) {
		printf("Essential Matrix: \n");
		showMatDoubleValue(E);
	}

	// Get T vector
	if (debug)
		printf("task: Get the R and T vectors\n");
	Vec3d D_t;
	Matx33d U_t, V_t;

	// Get SVD of E
	if (debug)
		printf("sub-task: Get SVD of E\n");

	SVD::compute(E, D_t, U_t, V_t, SVD::FULL_UV);
	D_t[0] = 1;
	D_t[1] = 1;
	D_t[2] = 0;
	Matx33d En = U_t*Matx33d::diag(D_t)*V_t;
	SVD::compute(En, D_t, U_t, V_t, SVD::FULL_UV);

	if (debug) {
		printf("E decomposed into U_t, D_t & V_t using SVD\n");
		printf("U_t:\n");
		cout << U_t << endl;
		printf("D_t: \n");
		cout << D_t << endl;
		printf("V_t: \n");
		cout << V_t << endl;
	}

	// Get the Translational vector
	if (debug)
		printf("sub-task: Get the Translational vector\n");
	Mat t1(3, 1, CV_64F);
	Mat t2(3, 1, CV_64F);
	for (int i = 0; i < 3; i++) {
		t1.at<double>(i, 0) = U_t(i, 2);
		t2.at<double>(i, 0) = -1*U_t(i, 2);
	}

	if (debug) {
		printf("Values of Translational vector are:\n");
		cout << t1 << endl;
		cout << t2 << endl;
	}

	// Calculate the Rotational Vector
	if (debug)
		printf("sub-task: Calculate the Rotational Vector\n");
	Matx33d R1_, R2_, V_t_transpose;
	Matx33d D1, D2;
	D1(0, 1) = -1;
	D1(1, 0) = 1;
	D2(0, 1) = 1;
	D2(1, 0) = -1;
	transpose(V_t, V_t_transpose);
	R1_ = U_t*D1*V_t.t();
	R2_ = U_t*D2*V_t.t();


	if (debug) {
		cout << "The Rotational Vectors are:\n";
		cout << R1_ << endl;
		cout << R2_ << endl;
	}

	// Calculate the Projection Matrix
	if (debug)
		printf("sub-task: Calculate the Projection Matrix\n");
	Mat P1_1 = Mat::zeros(3, 4, CV_64F);
	Mat P2_1 = Mat::zeros(3, 4, CV_64F);
	Mat P1_2 = Mat::zeros(3, 4, CV_64F);
	Mat P2_2 = Mat::zeros(3, 4, CV_64F);
	Mat P1_3 = Mat::zeros(3, 4, CV_64F);
	Mat P2_3 = Mat::zeros(3, 4, CV_64F);
	Mat P1_4 = Mat::zeros(3, 4, CV_64F);
	Mat P2_4 = Mat::zeros(3, 4, CV_64F);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j)
				P1_1.at<double>(i, j) = 1;
			P2_1.at<double>(i, j) = R1_(i, j);
		}
		P2_1.at<double>(i, 3) = t1.at<double>(i, 0);
	}
	P1_1 = K*P1_1;
	P2_1 = K*P2_1;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j)
				P1_2.at<double>(i, j) = 1;
			P2_2.at<double>(i, j) = R1_(i, j);
		}
		P2_2.at<double>(i, 3) = t2.at<double>(i, 0);
	}
	P1_2 = K*P1_2;
	P2_2 = K*P2_2;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j)
				P1_3.at<double>(i, j) = 1;
			P2_3.at<double>(i, j) = R2_(i, j);
		}
		P2_3.at<double>(i, 3) = t1.at<double>(i, 0);
	}
	P1_3 = K*P1_3;
	P2_3 = K*P2_3;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j)
				P1_4.at<double>(i, j) = 1;
			P2_4.at<double>(i, j) = R2_(i, j);
		}
		P2_4.at<double>(i, 3) = t2.at<double>(i, 0);
	}
	P1_4 = K*P1_4;
	P2_4 = K*P2_4;

	if (debug) {
		printf("The Camera Projection Matrix values are:\n");
		cout << "P1_1: \n" << P1_1 << endl;
		cout << "P2_1: \n" << P2_1 << endl;
		cout << "P1_2: \n" << P1_2 << endl;
		cout << "P2_2: \n" << P2_2 << endl;
		cout << "P1_3: \n" << P1_3 << endl;
		cout << "P2_3: \n" << P2_3 << endl;
		cout << "P1_4: \n" << P1_4 << endl;
		cout << "P2_4: \n" << P2_4 << endl;
	}


	waitKey(0);
	return 0;
}
