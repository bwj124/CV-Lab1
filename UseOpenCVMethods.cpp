#include<opencv2/opencv.hpp>

//#include<iostream>

using namespace cv;

int main(int argc, char** argv) {

	Mat image = imread("images/image.jpg");

	if (image.empty()) {

		printf("could not load image...\n");
		return -1;

	}

	//namedWindow("test_opencv_setup", 0);

	//imshow("test_opencv_srtup", image);

	Mat dst;
	Mat dstf;
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

	bilateralFilter(image, dst, 10, 100, 5);
	//imshow("dst", dst);
	filter2D(dst, dstf, dst.depth(), kernel);
	imwrite("images/1/image_bilateralFilter.jpg", dst);
	imwrite("images/1/image_bilateralFilter_filter.jpg", dstf);

	medianBlur(image, dst, 9);
	//imshow("dst", dst);
	filter2D(dst, dstf, dst.depth(), kernel);
	imwrite("images/1/image_medianBlur.jpg", dst);
	imwrite("images/1/image_medianBlur_filter.jpg", dstf);

	GaussianBlur(image, dst, Size(11, 11), 3, 3);
	filter2D(dst, dstf, dst.depth(), kernel);
	imwrite("images/1/image_GaussianBlur.jpg", dst);
	imwrite("images/1/image_GaussianBlur_filter.jpg", dstf);
	
	//waitKey(0);

	return 0;

}

