/*Underwater Object Detection involving color correction algorithm
Author : Sagnik Basu

The MIT License
Copyright (c) 2015 Avi Sagnik Basu
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double mul = 1;
int iter = 10; // iterations required for color correction
Point center;
Point critp;
Point lp;
float s1 = 0;
float s2 = 0;
int angle;
int scale = 1;
Mat rot_mat(2, 3, CV_32FC1);

int lowcount = 0;
int critpoint = 0;
int gl, rl, bl, bh = 255, rh = 255, gh = 255;
Mat img_bi, poly; // stores the binary image
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
vector<vector<Point>> polygons;

Mat correctGamma(Mat &img, double gamma) {
  double inverse_gamma = gamma;

  Mat lut_matrix(1, 256, CV_8UC1);
  uchar *ptr = lut_matrix.ptr();
  for (int i = 0; i < 256; i++)
    ptr[i] = (int)(pow((double)i / 255.0, inverse_gamma) * 255.0);

  Mat result;
  LUT(img, lut_matrix, result);

  return result;
}
// moments()
// VideoCapture front(0);

int main() {
  namedWindow("correction", WINDOW_NORMAL);
  namedWindow("Track", WINDOW_NORMAL);
  namedWindow("input", WINDOW_NORMAL);
  namedWindow("contour", WINDOW_NORMAL);
  createTrackbar("blue low", "Track", &bl, 255, NULL);
  createTrackbar("blue high", "Track", &bh, 255, NULL);
  createTrackbar("green low ", "Track", &gl, 255, NULL);
  createTrackbar("green high", "Track", &gh, 255, NULL);
  createTrackbar("red low", "Track", &rl, 255, NULL);
  createTrackbar("red high", "Track", &rh, 255, NULL);
  Mat img = imread("../resources/path.jpg", IMREAD_COLOR);
  imshow("input", img);
  Mat g_corr = correctGamma(img, 1);
  Mat final = Mat::zeros(img.size(), CV_8UC3);
  Mat imgg = g_corr.clone();

  // block for color correction
  for (int i = 0; i < iter; i++) {
    for (int p = 0; p < img.rows - img.rows / (64 * mul);
         p += img.rows / (64 * mul)) {
      for (int q = 0; q < img.cols - img.cols / (48 * mul);
           q += img.cols / (48 * mul)) {
        int bavg = 0;
        int gavg = 0;
        int ravg = 0;
        for (int i = p; i < p + img.rows / (64 * mul); i++) {
          for (int j = q; j < q + img.cols / (48 * mul); j++) {
            Vec3b color = g_corr.at<Vec3b>(i, j);
            int b = color[0];
            int g = color[1];
            int r = color[2];
            bavg += b;
            gavg += g;
            ravg += r;
          }
        }
        bavg = bavg * 64 * 48 * mul * mul / (img.rows * img.cols);
        gavg = gavg * 64 * 48 * mul * mul / (img.rows * img.cols);
        ravg = ravg * 64 * 48 * mul * mul / (img.rows * img.cols);
        int gg = 2 * gavg - (ravg + bavg);
        int rr = 2 * ravg - (gavg + bavg);
        for (int i = p; i < p + img.rows / (64 * mul); i++) {
          for (int j = q; j < q + img.cols / (48 * mul); j++) {
            Vec3b color = g_corr.at<Vec3b>(i, j);
            if (2 * color[2] >= color[1] + color[0] + iter + 15 &&
                ravg >= 45) //&&2*color[2]>color[1]+color[0])
            {
              color[1] = 255;
              color[0] = 255;
              color[2] = 255;
            } else {
              color[1] = 0;
              color[0] = 0;
              color[2] = 0;
            }

            imgg.at<Vec3b>(i, j) = color;
          }
        }
      }
    }
  }

  imshow("correction", imgg);

  waitKey(0);
}
