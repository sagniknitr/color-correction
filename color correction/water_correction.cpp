/*Underwater Object Detection involving color correction algorithm
Author : Sagnik Basu*/






#include<opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

double mul = 1;
int iter = 10;  //iterations required for color correction
Point center;
Point critp;
Point lp;
float s1 = 0; float s2 = 0;
int angle;
int scale = 1;
Mat rot_mat(2, 3, CV_32FC1);

int lowcount = 0;
int critpoint = 0;
int gl, rl, bl, bh=255, rh=255, gh=255;
Mat img_bi, poly;                      // stores the binary image
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
vector<vector<Point> > polygons;


Mat correctGamma(Mat& img, double gamma) {
    double inverse_gamma = gamma;

    Mat lut_matrix(1, 256, CV_8UC1);
    uchar * ptr = lut_matrix.ptr();
    for (int i = 0; i < 256; i++)
        ptr[i] = (int)(pow((double)i / 255.0, inverse_gamma) * 255.0);

    Mat result;
    LUT(img, lut_matrix, result);

    return result;
}
//moments()
//VideoCapture front(0);

int main()
{
    namedWindow("correction", CV_WINDOW_NORMAL);
    namedWindow("Track", CV_WINDOW_NORMAL);
    namedWindow("input", CV_WINDOW_NORMAL);
    namedWindow("contour", CV_WINDOW_NORMAL);
    //namedWindow("", CV_WINDOW_NORMAL);
    createTrackbar("blue low", "Track", &bl, 255, NULL);
    createTrackbar("blue high", "Track", &bh, 255, NULL);
    createTrackbar("green low ", "Track", &gl, 255, NULL);
    createTrackbar("green high", "Track", &gh, 255, NULL);
    createTrackbar("red low", "Track", &rl, 255, NULL);
    createTrackbar("red high", "Track", &rh, 255, NULL);
    Mat img = imread("/home/shaggy/line_auv/path.jpg", CV_LOAD_IMAGE_COLOR);
    imshow("input",img);
    Mat g_corr=correctGamma(img,1);
    Mat final = Mat::zeros(img.size(), CV_8UC3);
    Mat imgg = g_corr.clone();
    //cvtColor(img, img, COLOR_BGR2HSV);


    // block for color correction
    for (int i = 0; i < iter; i++)
    {
        for (int p = 0; p < img.rows - img.rows / (64 * mul); p += img.rows / (64 * mul))
        {
            for (int q = 0; q < img.cols - img.cols / (48 * mul); q += img.cols / (48 * mul))
            {
                int bavg = 0;
                int gavg = 0;
                int ravg = 0;
                for (int i = p; i < p + img.rows / (64 * mul); i++)
                {
                    for (int j = q; j < q + img.cols / (48 * mul); j++)
                    {
                        Vec3b color = g_corr.at<Vec3b>(i, j);
                        int b = color[0];
                        int g = color[1];
                        int r = color[2];
                        bavg += b;
                        gavg += g;
                        ravg += r;
                    }
                }
                bavg = bavg * 64 * 48 * mul*mul / (img.rows*img.cols);
                gavg = gavg * 64 * 48 * mul*mul / (img.rows*img.cols);
                ravg = ravg * 64 * 48 * mul*mul / (img.rows*img.cols);
                int gg = 2 * gavg - (ravg + bavg);
                int rr = 2 * ravg - (gavg + bavg);
                for (int i = p; i < p + img.rows / (64 * mul); i++)
                {
                    for (int j = q; j < q + img.cols / (48 * mul); j++)
                    {
                        Vec3b color = g_corr.at<Vec3b>(i, j);
                        /*if (2 * color[1] >= color[2] + color[0] + iter&&gavg >= 45)//&&2*color[1]>color[2]+color[0])
                        {
                            color[1] = 165;
                            color[0] = 0;
                            color[2] = 255;
                        }*/
                        if (2 * color[2] >= color[1] + color[0] + iter + 15 && ravg >= 45)//&&2*color[2]>color[1]+color[0])
                        {
                            color[1] = 255;
                            color[0] =255;
                            color[2] = 255;
                        }
                        else
                        {
                            color[1] =0;
                        color[0] =0;
                        color[2] =0;
}

                        imgg.at<Vec3b>(i, j) = color;
                    }
                }

            }
        }
    }
        //cvtColor(img, img_gray, CV_BGR2GRAY);
  //      while (1)
//        {

           // poly = Mat::zeros(img.size(), CV_8UC3);
            //inRange(img, Scalar(bl, 0, rl), Scalar(bh, 0, rh), img_bi);
           // bitwise_not(img_bi, img_bi);
         //   findContours(img_bi.clone(), contours,hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
          //  polygons.resize(contours.size());


          //  for (int i = 0; i < contours.size(); i++)
          //  {
           //     approxPolyDP(Mat(contours[i]), polygons[i], arcLength(Mat(contours[i]), true)*0.019, true);
           // }
           // for (int i = 0; i < polygons.size(); i++)
           // {
             //   Scalar color = Scalar(255, 255, 255);

               // drawContours(poly, polygons, i, color, 1, 8, hierarchy, 0, Point());
        //        for (int j = 0; j < polygons[i].size();j++)
          //      circle(poly, polygons[i][j], 4, Scalar(255, 255, 0), 1, 8, 0);
                //circle(poly, polygons[0][3], 5, Scalar(255, 0, 0), 1, 8, 0);
         //   }
          //  imshow("contour", poly);
          //  imshow("color correction", img);
        //    imshow("gamma correction", g_corr);
            //if (waitKey(32) == 27)
              //  break;
        //}
       //imshow("color correction", imgg);
       //medianBlur(imgg,imgg,5);
       for(int i=0;i<5;i++)
          {
       erode(imgg,imgg,Mat());
       }
      // erode(imgg,imgg,Mat());
       //erode(imgg,imgg,Mat());
        imshow("correction", imgg);

        waitKey(0);


}
