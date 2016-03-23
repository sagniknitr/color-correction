
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cmath>
//#include "ros/ros.h"
//#include "std_msgs/UInt16.h"
#include<iostream>

using namespace cv;
using namespace std;

Mat img;
//global variables
double mul = 1;
int iter = 12;  //iterations required for color correction
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

int camIndx=0;
VideoCapture front(camIndx);



double dst(Point2f p1, Point2f p2)
{
    double d = (p1.x - p2.x)*(p1.x - p2.x) +(p1.y - p2.y)*(p1.y - p2.y);
    double dist = pow(d, 0.5);
    return dist;
}

Mat cartesianRotate(Mat image, int method)
{
    if (method <0)
        transpose(image, image);
    //if(method == 1)
    //return image;
    Mat rotated = Mat(image.rows, image.cols, CV_MAKETYPE(CV_8U, image.channels()));
    //Vec3b value;
    //cout << image.size() << " " << rotated.size() <<  endl;
    flip(image, rotated, 1);
    /*for (int i = 0; i<rotated.cols; i++)
    for (int j = 0; j<rotated.rows; j++)
    rotated.at<Vec3b>(Point(i, j)) = image.at<Vec3b>(Point(image.cols - i - 1, j));*/

    if (method >0)
        transpose(rotated, rotated);
    return rotated;
}

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

int main()
{
    namedWindow("F.C.Image",CV_WINDOW_NORMAL);
    namedWindow("GammaCorr",CV_WINDOW_NORMAL);
    namedWindow("C.C.",CV_WINDOW_NORMAL);
    namedWindow("BuoyContours",CV_WINDOW_AUTOSIZE);
    //namedWindow();
    Mat ero,dil,front_orig;
    //enter the video input
    if(!front.isOpened())
{
        cout<<"Check Camera"<<"\n";
        return 0;
    }

    while(1){
       if(!front.read(front_orig))
       {
           continue;
       }
       Mat g_corr=correctGamma(front_orig,3.22);
       imshow("GammaCorr",g_corr);
       imshow("F.C.Image",front_orig);
    //enter the correction code
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
                        if (2 * color[1] >= color[2] + color[0] + iter&&gavg >= 45)//&&2*color[1]>color[2]+color[0])
                        {
                            color[1] = 255;
                            color[0] = 0;
                            color[2] = 255;
                        }
                        else if (2 * color[2] >= color[1] + color[0] + iter + 15 && ravg >= 65)//&&2*color[2]>color[1]+color[0])
                        {
                            color[1] = 0;
                            color[0] = 0;
                            color[2] = 255;
                        }

                       front_orig.at<Vec3b>(i, j) = color;
                    }
                }

            }
        }
    }
    imshow("C.C",front_orig);


    if(waitKey(10)==27)
        break;

    }
    return 0;
}



/*
int ballDetect(Mat img2)
{
    vector<Mat> imgf_base();

    int no_of_bases=5;
    //int j=0;
    // 1 = only water  2=yellow stick 3=green stick
    for(int i=0;i<no_of_bases;i++)
    {
        std::string s;
        std::stringstream out;
        out << i;
        s = out.str();
        Mat basef=imread(s+"f.jpg");
        //cvtColor( base, base, COLOR_BGR2HSV );
        imgf_base.push_back(basef);
    }

    Mat imgf_test=img2.clone();

    /// Using 50 bins for hue and 60 for saturation
       int h_bins = 50; int s_bins = 60;
       int histSize[] = { h_bins, s_bins };

       // hue varies from 0 to 179, saturation from 0 to 255
       float h_ranges[] = { 0, 180 };
       float s_ranges[] = { 0, 256 };

       const float* ranges[] = { h_ranges, s_ranges };

       // Use the o-th and 1-st channels
       int channels[] = { 0, 1 };

       /// Histograms
       MatND histograms_botf;
       vector<MatND> histograms_f(no_of_bases);

       /// Calculate the histograms for the HSV images
       for(int i=0;i<no_of_bases;i++)
       {
       //calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
      // normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
       calHist(&imgf_base(i),1,channels,Mat(),histograms_f(i),2,histSize,ranges,true,false);
       normalize(histograms(i),histograms(i),0,1,NORM_MINMAX,-1,Mat());

 }
       calHist(&img_test,1,channels,Mat(),histograms_bot,2,histSize,ranges,true,Mfalse);
       normalize(histograms_bot,histograms_bot,0,1,NORM_MINMAX,-1,Mat());

       int compare_method=1;double min_value;
       vector<double> compare_values(no_of_bases);

       for(int i=0;i<no_of_bases;i++)
      {
       //calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
      // normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );
  compare_values[i] = compareHist(histograms[i], histograms_bot, compare_method );
          if(min_value>compare_values[i])
          {
              min_value=compare_values[i];
              key =i;

          }

 }



       return key;

}
*/
