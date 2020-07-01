#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// From wikipedia:
//
// def RL_deconvolution(observed, psf, iterations):
//     # initial estimate is arbitrary - uniform 50% grey works fine
//     latent_est = 0.5*np.ones(observed.shape)
//     # create an inverse psf
//     psf_hat = psf[::-1,::-1]
//     # iterate towards ML estimate for the latent image
//     for i in np.arange(iterations):
//         est_conv      = cv2.filter2D(latent_est,-1,psf)
//         relative_blur = observed/est_conv;
//         error_est     = cv2.filter2D(relative_blur,-1,psf_hat)
//         latent_est    = latent_est * error_est
//     return latent_est

static int image_type;

Mat RL_deconvolution(Mat observed, Mat psf, int iterations) {

  Scalar grey;

  // Uniform grey starting estimation
  switch (image_type) {
  case CV_64FC1:
    grey = Scalar(0.5);
  case CV_64FC3:
    grey = Scalar(0.5, 0.5, 0.5);
  }
  Mat latent_est = Mat(observed.size(), image_type, grey);

  // Flip the point spread function (NOT the inverse)
  Mat psf_hat = Mat(psf.size(), CV_64FC1);
  int psf_row_max = psf.rows - 1;
  int psf_col_max = psf.cols - 1;
  for (int row = 0; row <= psf_row_max; row++) {
    for (int col = 0; col <= psf_col_max; col++) {
      psf_hat.at<double>(psf_row_max - row, psf_col_max - col) =
          psf.at<double>(row, col);
    }
  }

  Mat est_conv;
  Mat relative_blur;
  Mat error_est;

  // Iterate
  for (int i = 0; i < iterations; i++) {

    filter2D(latent_est, est_conv, -1, psf);

    // Element-wise division
    relative_blur = observed.mul(1.0 / est_conv);

    filter2D(relative_blur, error_est, -1, psf_hat);

    // Element-wise multiplication
    latent_est = latent_est.mul(error_est);
  }

  return latent_est;
}

int main() {

  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap("../resources/buoy.avi");

  // Check if camera opened successfully
  if (!cap.isOpened()) {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  while (1) {

    Mat original_image;
    // Capture frame-by-frame
    cap >> original_image;

    // If the frame is empty, break immediately
    if (original_image.empty())
      break;

    int iterations = 2;

    int num_channels = original_image.channels();
    switch (num_channels) {
    case 1:
      image_type = CV_64FC1;
      break;
    case 3:
      image_type = CV_64FC3;
      break;
    default:
      return -2;
    }

    // This is a hack, assumes too much
    int divisor;
    switch (original_image.elemSize() / num_channels) {
    case 1:
      divisor = 255;
      break;
    case 2:
      divisor = 65535;
      break;
    default:
      return -3;
    }

    // From here on, use 64-bit floats
    // Convert original_image to float
    Mat float_image;
    original_image.convertTo(float_image, image_type);
    float_image *= 1.0 / divisor;
    namedWindow("Float", CV_WINDOW_AUTOSIZE);
    imshow("Float", float_image);

    // Calculate a gaussian blur psf.
    double sigma_row = 9.0;
    double sigma_col = 5.0;
    int psf_size = 5;
    double mean_row = 0.0;
    double mean_col = psf_size / 2.0;
    double sum = 0.0;
    double temp;
    Mat psf = Mat(Size(psf_size, psf_size), CV_64FC1, 0.0);

    for (int j = 0; j < psf.rows; j++) {
      for (int k = 0; k < psf.cols; k++) {
        temp = exp(-0.5 * (pow((j - mean_row) / sigma_row, 2.0) +
                           pow((k - mean_col) / sigma_col, 2.0))) /
               (2 * M_PI * sigma_row * sigma_col);
        sum += temp;
        psf.at<double>(j, k) = temp;
      }
    }

    // Normalise the psf.
    for (int row = 0; row < psf.rows; row++) {
      // cout << row << " ";
      for (int col = 0; col < psf.cols; col++) {
        psf.at<double>(row, col) /= sum;
        // cout << psf.at<double>(row, col) << " ";
      }
      // cout << "\n";
    }

    // Blur the float_image with the psf.
    Mat blurred_float;
    blurred_float = float_image.clone();
    filter2D(float_image, blurred_float, -1, psf);
    namedWindow("BlurredFloat", CV_WINDOW_AUTOSIZE);
    imshow("BlurredFloat", blurred_float);

    Mat estimation = RL_deconvolution(blurred_float, psf, iterations);
    namedWindow("Estimation", CV_WINDOW_AUTOSIZE);
    imshow("Estimation", estimation);
    // Press  ESC on keyboard to exit
    char c = (char)waitKey(25);
    if (c == 27)
      break;
  }

  // When everything done, release the video capture object
  cap.release();
  destroyAllWindows();
}
