/**
 * @file    test.cpp
 * @brief   test file
 * @author  ZYP
*/

#include <gflags/gflags.h>
#include "yolox.hpp"
#include "color.hpp"

DEFINE_string(input_model, "./data/yolox_nano_ti_lite_float32.tflite", "input model");
DEFINE_string(input_image, "./data/classic_image.jpg", "input image");
DEFINE_string(output_image, "./data/classic_image_detect.jpg", "output image");
DEFINE_string(label_path, "./data/labels.txt", "labels file");
DEFINE_bool(quantify, false, "whether use quant-model");
DEFINE_int32(resolution, 416, "default image resolution");
DEFINE_double(score_threshold, 0.35, "default score threshold");
DEFINE_double(nms_threshold, 0.7, "default nms threshold");

using namespace Aidlux::Aidlite;

std::map<int, std::string> label_map_;

bool readLabelFile(const std::string & label_path)
{
  std::ifstream label_file(label_path);
  if (!label_file.is_open()) {
    LOG(ERROR) << "Could not open label file. " << label_path << std::endl;;
    return false;
  }
  int label_index{};
  std::string label;
  while (getline(label_file, label)) {
    std::transform(
      label.begin(), label.end(), label.begin(), [](auto c) { return std::toupper(c); });
    label_map_.insert({label_index, label});
    ++label_index;
  }
  return true;
}

int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;

    auto yolox = std::make_shared<YoloX>(FLAGS_input_model, FLAGS_resolution, FLAGS_score_threshold, FLAGS_nms_threshold, FLAGS_quantify);

    cv::Mat input_img = cv::imread(FLAGS_input_image);
    ObjectArray objects;
    yolox->doInference(input_img, objects);

    readLabelFile(FLAGS_label_path);

    for (const auto & object : objects) {
        // color
        float* color_f = _COLORS[object.type];
        std::vector<int> color = { static_cast<int>(color_f[0] * 255), static_cast<int>(color_f[1] * 255), static_cast<int>(color_f[2] * 255) };

        // text
        std::string text = label_map_[object.type] + ":" + std::to_string(object.score * 100) + "%";
        cv::Scalar txt_color = ((color_f[0] + color_f[1] + color_f[2]) > 0.5) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
        int font = cv::FONT_HERSHEY_SIMPLEX;
        int baseline = 0;
        cv::Size txt_size = cv::getTextSize(text, font, 0.4, 1, &baseline);
        
        const auto left = object.x_offset;
        const auto top = object.y_offset;
        const auto right = std::clamp(left + object.width, 0, input_img.cols);
        const auto bottom = std::clamp(top + object.height, 0, input_img.rows);
        cv::rectangle(
        input_img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3, 8, 0);

        // text bg
        std::vector<int> txt_bk_color = { static_cast<int>(color_f[0] * 255 * 0.7), static_cast<int>(color_f[1] * 255 * 0.7), static_cast<int>(color_f[2] * 255 * 0.7) };
        cv::rectangle(
            input_img,
            cv::Point(left, top + 1),
            cv::Point(left + txt_size.width + 1, top + int(1.5 * txt_size.height)),
            cv::Scalar(txt_bk_color[0], txt_bk_color[1], txt_bk_color[2]),
            -1
        );

        cv::putText(input_img, text, cv::Point(left, top + txt_size.height), font, 0.4, txt_color, 1);
    }

    // output image
    cv::imwrite(FLAGS_output_image, input_img);

    return 0;
}