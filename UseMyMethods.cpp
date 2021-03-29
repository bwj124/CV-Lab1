#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

using namespace cv;
using namespace std;


// 获取中值
double sort_color(vector<double> list);

/*
* 中值滤波
* src是输入图像
* dst是输出图像
* size是滤波器的大小，只能够为奇数
*/
void myMedianBlur(const Mat src, Mat& dst, int size); 

double** getGaussionArray(int arr_size, double sigma);

/*
* 高斯滤波
* src是输入图像
* dst是输出图像
* size表示滤波器的大小
* sigma表示高斯函数方差
*/
void myGaussianBlur(Mat src, Mat& dst, int size, double sigma);

/* 计算空间权值 */
double** get_space_Array(int _size, int channels, double sigmas);

/* 计算相似度权值 */
double* get_color_Array(int _size, int channels, double sigmar);

/* 双边 扫描计算 */
void doBialteral(cv::Mat* _src, int N, double* _colorArray, double** _spaceArray);

/* 
* 双边滤波
* 
*/
void myBialteralFilter(cv::Mat* src, cv::Mat* dst, int N, double sigmas, double sigmar);

int main() {
	Mat image = imread("images/image.jpg");
	if (image.empty())
	{
		printf("could not load image\n");
		return -1;
	}

	Mat dst;
	Mat dstf;
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

	cout << "Program is running, please wait for a moment." << endl;

	//myGaussianBlur(image, dst, 3, 20);
	//imwrite("images/2/image_GaussianBlur.jpg", dst);
	//filter2D(dst, dstf, dst.depth(), kernel);
	//imwrite("images/2/image_GaussianBlur_filter.jpg", dstf);

	myMedianBlur(image, dst, 3);
	imwrite("images/2/image_MedianBlur.jpg", dst);
	filter2D(dst, dstf, dst.depth(), kernel);
	imwrite("images/2/image_MedianBlur_filter.jpg", dstf);

	//myBialteralFilter(&image, &dst, 25, 12.5, 50);
	//imwrite("images/2/image_BialteralFilter.jpg", dst);
	//filter2D(dst, dstf, dst.depth(), kernel);
	//imwrite("images/2/image_BialteralFilter_filter.jpg", dstf);

	waitKey(0);

	return 0;
}


double** getGaussionArray(int arr_size, double sigma)
{
	int i, j;
	double** array = new double* [arr_size];
	for (i = 0; i < arr_size; i++) {
		array[i] = new double[arr_size];
	}
	int center_i, center_j;
	center_i = center_j = arr_size / 2;
	double pi = 3.141592653589793;
	double sum = 0.0f;
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] =
				exp(-(1.0f) * (((i - center_i) * (i - center_i) + (j - center_j) * (j - center_j)) /
					(2.0f * sigma * sigma)));
			sum += array[i][j];
		}
	}
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] /= sum;
		}
	}
	return array;
}

void myGaussianBlur(Mat src, Mat& dst, int size, double sigma) {
	if (!src.data) return;
	double** arr;
	Mat tmp(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				arr = getGaussionArray(size, sigma);
				tmp.at<Vec3b>(i, j)[0] = 0;
				tmp.at<Vec3b>(i, j)[1] = 0;
				tmp.at<Vec3b>(i, j)[2] = 0;
				for (int x = 0; x < 3; ++x) {
					for (int y = 0; y < 3; ++y) {
						tmp.at<Vec3b>(i, j)[0] += arr[x][y] * src.at<Vec3b>(i + 1 - x, j + 1 - y)[0];
						tmp.at<Vec3b>(i, j)[1] += arr[x][y] * src.at<Vec3b>(i + 1 - x, j + 1 - y)[1];
						tmp.at<Vec3b>(i, j)[2] += arr[x][y] * src.at<Vec3b>(i + 1 - x, j + 1 - y)[2];
					}
				}
			}
		}
	tmp.copyTo(dst);
}

double sort_color(vector<double> list) {
	nth_element(list.begin(), list.begin() + (list.size() / 2) + 1, list.end());
	return list[(list.size() / 2)];
}

void myMedianBlur(const Mat src, Mat& dst, int size) {
	if ((size % 2) == 0) {
		return;
	}
	if (!src.data) return;
	Mat _dst(src.size(), src.type());
	int bide = size / 2;
	int kernel_size = size * size;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if ((i - bide) >= 0 && (j - bide) >= 0 && (i + bide) < src.rows && (j + bide) < src.cols) {
				vector<double>kernel_value_0(kernel_size);
				vector<double>kernel_value_1(kernel_size);
				vector<double>kernel_value_2(kernel_size);
				for (int kernel_row = i - bide; kernel_row <= i + bide; kernel_row++) {
					for (int kernel_col = j - bide; kernel_col <= j + bide; kernel_col++) {
						kernel_value_0.push_back(src.at<Vec3b>(kernel_row, kernel_col)[0]);
						kernel_value_1.push_back(src.at<Vec3b>(kernel_row, kernel_col)[1]);
						kernel_value_2.push_back(src.at<Vec3b>(kernel_row, kernel_col)[2]);
					}
				}
				_dst.at<Vec3b>(i, j)[0] = sort_color(kernel_value_0);
				_dst.at<Vec3b>(i, j)[1] = sort_color(kernel_value_1);
				_dst.at<Vec3b>(i, j)[2] = sort_color(kernel_value_2);
			}
		}
	}
	_dst.copyTo(dst);
}

/* 计算空间权值 */
double** get_space_Array(int _size, int channels, double sigmas)
{
	int i, j;
	double** _spaceArray = new double* [_size + 1];   //多一行，最后一行的第一个数据放总值
	for (i = 0; i < _size + 1; i++) {
		_spaceArray[i] = new double[_size + 1];
	}
	int center_i, center_j;
	center_i = center_j = _size / 2;
	_spaceArray[_size][0] = 0.0f;
	for (i = 0; i < _size; i++) {
		for (j = 0; j < _size; j++) {
			_spaceArray[i][j] =
				exp(-(1.0f) * (((i - center_i) * (i - center_i) + (j - center_j) * (j - center_j)) /
					(2.0f * sigmas * sigmas)));
			_spaceArray[_size][0] += _spaceArray[i][j];
		}
	}
	return _spaceArray;
}

/* 计算相似度权值 */
double* get_color_Array(int _size, int channels, double sigmar)
{
	int n;
	double* _colorArray = new double[255 * channels + 2];
	double wr = 0.0f;
	_colorArray[255 * channels + 1] = 0.0f;
	for (n = 0; n < 255 * channels + 1; n++) {
		_colorArray[n] = exp((-1.0f * (n * n)) / (2.0f * sigmar * sigmar));
		_colorArray[255 * channels + 1] += _colorArray[n];
	}
	return _colorArray;
}

/* 双边 扫描计算 */
void doBialteral(cv::Mat* _src, int N, double* _colorArray, double** _spaceArray)
{
	int _size = (2 * N + 1);
	cv::Mat temp = (*_src).clone();
	for (int i = 0; i < (*_src).rows; i++) {
		for (int j = 0; j < (*_src).cols; j++) {
			if (i > (_size / 2) - 1 && j > (_size / 2) - 1 &&
				i < (*_src).rows - (_size / 2) && j < (*_src).cols - (_size / 2)) {
				double sum[3] = { 0.0,0.0,0.0 };
				int x, y, values;
				double space_color_sum = 0.0f;
				for (int k = 0; k < _size; k++) {
					for (int l = 0; l < _size; l++) {
						x = i - k + (_size / 2);  
						y = j - l + (_size / 2);   
						values = abs((*_src).at<cv::Vec3b>(i, j)[0] + (*_src).at<cv::Vec3b>(i, j)[1] + (*_src).at<cv::Vec3b>(i, j)[2]
							- (*_src).at<cv::Vec3b>(x, y)[0] - (*_src).at<cv::Vec3b>(x, y)[1] - (*_src).at<cv::Vec3b>(x, y)[2]);
						space_color_sum += (_colorArray[values] * _spaceArray[k][l]);
					}
				}
				for (int k = 0; k < _size; k++) {
					for (int l = 0; l < _size; l++) {
						x = i - k + (_size / 2);
						y = j - l + (_size / 2);
						values = abs((*_src).at<cv::Vec3b>(i, j)[0] + (*_src).at<cv::Vec3b>(i, j)[1] + (*_src).at<cv::Vec3b>(i, j)[2]
							- (*_src).at<cv::Vec3b>(x, y)[0] - (*_src).at<cv::Vec3b>(x, y)[1] - (*_src).at<cv::Vec3b>(x, y)[2]);
						for (int c = 0; c < 3; c++) {
							sum[c] += ((*_src).at<cv::Vec3b>(x, y)[c]
								* _colorArray[values]
								* _spaceArray[k][l])
								/ space_color_sum;
						}
					}
				}
				for (int c = 0; c < 3; c++) {
					temp.at<cv::Vec3b>(i, j)[c] = sum[c];
				}
				if ((i % 100) == 0 && j == 10)
				{
					cout << i << endl;
				}
				if (i > 5000000)
				{
					return;
				}
			}
		}
	}
	(*_src) = temp.clone();

	return;
}

void myBialteralFilter(cv::Mat* src, cv::Mat* dst, int N, double sigmas, double sigmar)
{
	*dst = (*src).clone();
	int _size = 2 * N + 1;
	int channels = (*dst).channels();
	double* _colorArray = NULL;
	double** _spaceArray = NULL;
	_colorArray = get_color_Array(_size, channels, sigmar);
	_spaceArray = get_space_Array(_size, channels, sigmas);
	doBialteral(dst, N, _colorArray, _spaceArray);

	return;
}