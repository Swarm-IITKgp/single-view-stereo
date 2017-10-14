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
		D.at<double>(0, i) = temp;
	}
}

void scale_projection_matrix(Mat &P) {
	float scale = 1.0;
	for (int i = 0; i < P.cols; i++) {
		scale = P.at<double>(3, i);
		if (isnan(scale))
			continue;

		for (int j = 0; j < P.rows; j++) {
			P.at<double>(j, i) = P.at<double>(j, i) / scale;
		}
	}
}

void createMat(Mat &X, vector<Point> &v) {
	for (int i = 0; i < v.size(); i++) {
		X.at<double>(0, i) = v[i].x;
		X.at<double>(1, i) = v[i].y;
	}
}

int is_good_solution(Mat &P, Mat &t) {
	int is_good = 1;
	for (int i = 0; i < P.cols; i++) {
		if (isnan(P.at<double>(2, i)))
			continue;
		if (P.at<double>(2, i) < t.at<double>(2, 0) || P.at<double>(2, i) < 0)
			is_good = 0;
	}
	return is_good;
}

void write_time(FILE *fp, char *task, int value) {
	fprintf(fp, "%s: %d\n", task, value);
}

void undistort_rot( InputArray _src, OutputArray _dst, InputArray _cameraMatrix,
                    InputArray _distCoeffs, InputArray _newCameraMatrix, InputArray Rot)
{
	Mat src = _src.getMat(), cameraMatrix = _cameraMatrix.getMat();
    Mat distCoeffs = _distCoeffs.getMat(), newCameraMatrix = _newCameraMatrix.getMat();

    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    int stripe_size0 = std::min(std::max(1, (1 << 12) / std::max(src.cols, 1)), src.rows);
    Mat map1(stripe_size0, src.cols, CV_16SC2), map2(stripe_size0, src.cols, CV_16UC1);

    Mat_<double> A, Ar, I = Mat_<double>::eye(3,3);

    cameraMatrix.convertTo(A, CV_64F);
    if( !distCoeffs.empty() )
        distCoeffs = Mat_<double>(distCoeffs);
    else
    {
        distCoeffs.create(5, 1, CV_64F);
        distCoeffs = 0.;
    }

    if( !newCameraMatrix.empty() )
        newCameraMatrix.convertTo(Ar, CV_64F);
    else
        A.copyTo(Ar);

    double v0 = Ar(1, 2);
    for( int y = 0; y < src.rows; y += stripe_size0 )
    {
        int stripe_size = std::min( stripe_size0, src.rows - y );
        Ar(1, 2) = v0 - y;
        Mat map1_part = map1.rowRange(0, stripe_size),
            map2_part = map2.rowRange(0, stripe_size),
            dst_part = dst.rowRange(y, y + stripe_size);

        initUndistortRectifyMap( A, distCoeffs, Rot, Ar, Size(src.cols, stripe_size),
                                 map1_part.type(), map1_part, map2_part );
        remap( src, dst_part, map1_part, map2_part, INTER_LINEAR, BORDER_CONSTANT );
    }
}

int main() {
	FILE *fp;
	char *debug = getenv("DEBUG");
	time_t initial_t, final_t;

	fp = fopen("time.log", "w");
	// Read the images as grayscale
	Mat left = imread("left.jpg", 0);
	Mat right = imread("right.jpg", 0);

	if (debug) {
		namedWindow("Left", WINDOW_AUTOSIZE);
		namedWindow("Right", WINDOW_AUTOSIZE);
		imshow("Left", left);
		imshow("Right", right);
	}

	// Detect the keypoints using the ORB detector
	if (debug)
		printf("task: Detect the keypoints using SURF detector\n");
	initial_t = time(NULL);
	int minHessian = 400;
	vector<KeyPoint> left_keypoints, right_keypoints;
	Mat left_descriptors, right_descriptors;
	Ptr<ORB> orb = ORB::create();

	orb->detectAndCompute(left, Mat(), left_keypoints, left_descriptors, false);
	orb->detectAndCompute(right, Mat(), right_keypoints, right_descriptors, false);
	final_t = time(NULL);
	write_time(fp, "Detect the keypoints using ORB Detector", final_t-initial_t);

	if (debug) {
		Mat img_keypoints_left, img_keypoints_right;
		drawKeypoints(left, left_keypoints, img_keypoints_left, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		drawKeypoints(right, right_keypoints, img_keypoints_right, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		imshow("Left Keypoints", img_keypoints_left);
		imshow("Right keypoints", img_keypoints_right);
	}

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
	initial_t = time(NULL);
	FlannBasedMatcher matcher;
	vector<DMatch> matches;

	matcher.match(left_descriptors, right_descriptors, matches);
	final_t = time(NULL);
	write_time(fp, "Matching descriptor vectors using FLANN matcher", final_t - initial_t);

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
	initial_t = time(NULL);
	vector<DMatch> good_matches;
	for (int i = 0; i < left_descriptors.rows; i++)
		if (matches[i].distance <= max(10*min_dist, 0.02))
			good_matches.push_back(matches[i]);
	final_t = time(NULL);
	write_time(fp, "Get the good matches ie. that have only little error value", final_t - initial_t);

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
	initial_t = time(NULL);
	vector<Point2f> left_imp_points, right_imp_points;

	for (size_t i = 0; i < good_matches.size(); i++) {
		left_imp_points.push_back(left_keypoints[good_matches[i].queryIdx].pt);
		right_imp_points.push_back(right_keypoints[good_matches[i].trainIdx].pt);
	}
	final_t = time(NULL);
	write_time(fp, "Get the vector of points which have a corresponding spot in both", final_t - initial_t);

	// Finding The Fundamental Matrix
	if (debug)
		printf("task: Finding the Fundamental Matrix\n");
	initial_t = time(NULL);
	Mat F = findFundamentalMat(Mat(left_imp_points), Mat(right_imp_points), CV_FM_RANSAC);
	final_t = time(NULL);
	write_time(fp, "Finding the Fundamental Matrix", final_t - initial_t);

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
	Mat D = Mat::zeros(1, 5, CV_64F);
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
	//D_t[0] = 1;
	//D_t[1] = 1;
	//D_t[2] = 0;
	// Matx33d En = U_t*Matx33d::diag(D_t)*V_t;
	// SVD::compute(En, D_t, U_t, V_t, SVD::FULL_UV);

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
	D1(2, 2) = 1;
	D2(0, 1) = 1;
	D2(1, 0) = -1;
	D2(2, 2) = 1;
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

	// Get the corresponding 3D points for each projection matrix set
	if (debug)
		printf("task: Get the corresponding 3D for each projection matrix set\n");

	Mat points_3D_1, points_3D_2, points_3D_3, points_3D_4;
	Mat left_imp_points_mat = Mat(2, left_imp_points.size(), CV_64F);
	Mat right_imp_points_mat = Mat(2, right_imp_points.size(), CV_64F);
	Mat P1, P2, R, t;
	triangulatePoints(P1_1, P2_1, left_imp_points_mat, right_imp_points_mat, points_3D_1);
	triangulatePoints(P1_2, P2_2, left_imp_points_mat, right_imp_points_mat, points_3D_2);
	triangulatePoints(P1_3, P2_3, left_imp_points_mat, right_imp_points_mat, points_3D_3);
	triangulatePoints(P1_4, P2_4, left_imp_points_mat, right_imp_points_mat, points_3D_4);

	scale_projection_matrix(points_3D_1);
	scale_projection_matrix(points_3D_2);
	scale_projection_matrix(points_3D_3);
	scale_projection_matrix(points_3D_4);

	if (debug) {
		printf("The points_3D_1 mat is:\n");
		cout << points_3D_1 << endl;
		printf("The points_3D_2 mat is:\n");
		cout << points_3D_2 << endl;
		printf("The points_3D_3 mat is:\n");
		cout << points_3D_3 << endl;
		printf("The points_3D_4 mat is:\n");
		cout << points_3D_4 << endl;
	}

            bool found_solution=false;

	if (is_good_solution(points_3D_1, t1)) {
		printf("Selecting solution 1\n");
		P1 = Mat(P1_1);
		P2 = Mat(P2_2);
		R = Mat(R1_);
		t = Mat(t1);
                          found_solution=true;
	}
	if (is_good_solution(points_3D_2, t2)) {
		printf("Selecting solution 2\n");
		P1 = Mat(P1_2);
		P2 = Mat(P2_2);
		R = Mat(R1_);
		t = Mat(t2);
                          found_solution=true;
	}
	if (is_good_solution(points_3D_3, t1)) {
		printf("Selecting solution 3\n");
		P1 = Mat(P1_3);
		P2 = Mat(P2_3);
		R = Mat(R2_);
		t = Mat(t1);
                          found_solution=true;
	}
	if (is_good_solution(points_3D_4, t2)) {
		printf("Selecting solution 4\n");
		P1 = Mat(P1_4);
		P2 = Mat(P2_4);
		R = Mat(R2_);
		t = Mat(t2);
                          found_solution=true;
	}

            if (!found_solution)
            {
                cerr<<"Unable to determine rotational matrix."<<endl;
                return -1;
            }

	// Get the rectification parameters
	if (debug)
		printf("task: Get the rectification parameters\n");
	Rect validRoi[2];
	Mat R1, R2, t_rectified, P1_rectified, P2_rectified, Q;
	Size s = left.size();
	stereoRectify(K, D, K, D, s, R, t, R1, R2, P1_rectified, P2_rectified, Q, CALIB_ZERO_DISPARITY, 1, s, &validRoi[0], &validRoi[1]);

	if (debug) {
		cout << "The rectified R1 is: " << endl;
		cout << R1 << endl;
		cout << "The rectified R2 is: " << endl;
		cout << R2 << endl;
		cout << "The P1_rectified is: " << endl;
		cout << P1_rectified << endl;
		cout << "The P2_rectified is: " << endl;
		cout << P2_rectified << endl;
		cout << "The Q value is: " << endl;
		cout << Q << endl;
	}

	// Rectify the initial left and right images
	if (debug)
		printf("task: Rectify the initial left and right images");
	Mat left_undistorted, right_undistorted;
	Mat K_new;
	undistort_rot(left, left_undistorted, K, D, K_new, R);
	undistort_rot(right, right_undistorted, K, D, K_new, R);

	if (debug) {
		imshow("Left Undistorted image", left_undistorted);
		imshow("Right Undistorted image", right_undistorted);
	}

	// Get the disparity map using StereoBM
	Mat disparity_map;
	Ptr<StereoBM> sbm = StereoBM::create(16*5, 21);

	sbm->compute(left_undistorted, right_undistorted, disparity_map);

	// Apply Bilateral Filter
	Mat disparity_map_filtered_, disparity_map_filtered;
	disparity_map.convertTo(disparity_map_filtered_, CV_8UC1);
	for (int i = 1; i < 5; i = i + 2) {
		bilateralFilter(disparity_map_filtered_, disparity_map_filtered, i, i*2, i/2);
	}

	if (debug)
		imshow("Disparity Map", disparity_map_filtered);

	imwrite("disparity_map.jpg", disparity_map_filtered);
	disparity_map_filtered.convertTo(disparity_map, CV_32F);
	Mat depth_map;
	reprojectImageTo3D(disparity_map, depth_map, Q, true);

	waitKey(0);
	return 0;
}
