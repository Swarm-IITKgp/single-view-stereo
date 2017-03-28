#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
   char *debug = getenv("DEBUG");

	// Read the images as grayscale
	Mat left = imread("left.jpg", 0);
	Mat right = imread("right.jpg", 0);

	// Show the images
	if (debug) {
		namedWindow("Left", WINDOW_AUTOSIZE);
		namedWindow("Right", WINDOW_AUTOSIZE);
		imshow("Left", left);
		imshow("Right", right);
	}


	waitKey(0);
	return 0;
}
