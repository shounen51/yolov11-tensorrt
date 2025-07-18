#ifndef Tools_ColorClassifier_h_
#define Tools_ColorClassifier_h_

#include <vector>
#include <map>
#include <queue>
#include <math.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"


typedef struct Color_dataType
{
    Color_dataType(unsigned char v0=0, unsigned char v1=0, unsigned char v2=0);
    ~Color_dataType() = default;
    unsigned char val[3];
}Color_t;

class ColorRange
{
public:
    ColorRange();
    ColorRange(Color_t _range_min, Color_t _range_max, unsigned char colorLabel=0);
    ~ColorRange();
    bool empty();
    bool isInside(Color_t val);
    bool isInside(unsigned char val[3]);
    Color_t range_min, range_max;
    unsigned char label;
    std::vector<int> intersectlabelIndexList;
};

ColorRange& operator &= (ColorRange& a, const ColorRange& b);
ColorRange operator & (const ColorRange& a, const ColorRange& b);

static const std::vector<std::string> ColorLabelsString = {
    "unknow", "black", "gray", "white", // IsGrayscale
    "red", "orange", "brown", "yellow", "green", "cyan", "blue", "purple" // IsColor
};

enum ColorLabels : unsigned char
{
    // IsGrayscale = 7, IsColor = 248, //~IsGrayscale,
    color_unknow=0, color_black, color_gray, color_white,                                                                 // IsGrayscale
    color_red, color_orange, color_brown, color_yellow, color_green, color_cyan, color_blue, color_purple   // IsColor
};

class HSVColorClassifier
{
public:
    HSVColorClassifier();
    HSVColorClassifier(std::vector<ColorRange> colorRangeMap);
    ~HSVColorClassifier();

    std::vector<unsigned char> classify(Color_t color);
    std::vector<std::vector<unsigned char>> classify(unsigned char *array, unsigned int size, int *maxLabelCount=(int *)0);
    std::vector<unsigned char> classifyStatistics(cv::Mat image, int sampleNumber, int cvtColorCode, cv::Mat mask=cv::Mat(), bool usePercent=true);

    void setNewColorRange(std::vector<ColorRange> colorRangeMap);
    void setDefaultColorRange();
    int getMaxColorLabelNumber() { return maxColorLabelNum; }
private:

    void checkRangeHaveIntersect();

    std::vector<ColorRange> colorRangeMap;
    int maxColorLabelNum;
};





#endif // Tools_ColorClassifier_h_