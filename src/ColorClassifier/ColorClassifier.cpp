#include "ColorClassifier.h"


Color_dataType::Color_dataType(unsigned char v0, unsigned char v1, unsigned char v2)
{
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
}

ColorRange::ColorRange()
{
    range_min = Color_dataType();
    range_max = Color_dataType();
    label = 0;
}
ColorRange::ColorRange(Color_t _range_min, Color_t _range_max, unsigned char colorLabel)
{
    for (int i = 0; i < 3; i++)
    {
        range_max.val[i] = (_range_max.val[i] > _range_min.val[i]) ? _range_max.val[i] : _range_min.val[i];
        range_min.val[i] = (_range_max.val[i] > _range_min.val[i]) ? _range_min.val[i] : _range_max.val[i];
    }

    label = colorLabel;
}

ColorRange::~ColorRange()
{
    intersectlabelIndexList.clear();
}

bool ColorRange::empty()
{
    int sum = 0;
    for (int i = 0; i < 3; i++)
    {
        sum += abs(range_max.val[i] - range_min.val[i]);
    }
    return (sum == 0);
}

bool ColorRange::isInside(Color_t color)
{
    bool ret = true;
    for (int i = 0; i < 3 && ret; i++)
    {
        ret = (range_min.val[i] <= color.val[i] && color.val[i] <= range_max.val[i]) ? ret : false;
    }
    return ret;
}
bool ColorRange::isInside(unsigned char color[3])
{
    bool ret = true;
    for (int i = 0; i < 3 && ret; i++)
    {
        ret = (range_min.val[i] <= color[i] && color[i] < range_max.val[i]) ? ret : false;
    }
    return ret;
}

ColorRange& operator &= (ColorRange& a, const ColorRange& b)
{
    bool haveIntersection = true;
    Color_t R_min, R_max;
    for (int i = 0; i < 3 && haveIntersection; i++)
    {
        unsigned char _min = a.range_min.val[i] > b.range_min.val[i] ? a.range_min.val[i] : b.range_min.val[i];
        unsigned char _max = a.range_max.val[i] < b.range_max.val[i] ? a.range_max.val[i] : b.range_max.val[i];

        if (_max - _min <= 0)
        {
            haveIntersection = false;
            continue;
        }
        R_min.val[i] = _min;
        R_max.val[i] = _max;
    }

    if (!haveIntersection)
    {
        a = ColorRange(Color_t(), Color_t());
        return a;
    }

    a.range_min = R_min;
    a.range_max = R_max;
    a.label =  (a.label == b.label) ? a.label : 0;

    return a;
}

ColorRange operator & (const ColorRange& a, const ColorRange& b)
{
    ColorRange c = a;
    return c &= b;
}



HLSColorClassifier::HLSColorClassifier()
{

}
HLSColorClassifier::HLSColorClassifier(std::vector<ColorRange> colorRangeMap)
{
    this->colorRangeMap = colorRangeMap;
    checkRangeHaveIntersect();
}
HLSColorClassifier::~HLSColorClassifier()
{

}

std::vector<unsigned char> HLSColorClassifier::classify(Color_t color)
{
    std::vector<unsigned char> output_colorLabels;
    std::vector<int> candidateColorIdxList;
    bool notFindColor = true;

    // printf("color %d %d %d\n", color.val[0], color.val[1], color.val[2]);
    for (int map_i = 0; map_i < colorRangeMap.size() && notFindColor; map_i++)
    {
        auto *it = &colorRangeMap[map_i];
        if (!it->isInside(color))
            continue;

        candidateColorIdxList.insert(candidateColorIdxList.end(),
                        it->intersectlabelIndexList.begin(), it->intersectlabelIndexList.end());
        output_colorLabels.push_back(it->label);
        notFindColor = false;
    }

    while (candidateColorIdxList.size()>0)
    {
        int map_i = candidateColorIdxList[0];
        candidateColorIdxList.erase(candidateColorIdxList.begin());
        auto *it = &colorRangeMap[map_i];
        bool outExist = true;

        // printf("while() label:%d\n", it->label);
        // for (auto out_it = output_colorLabels.begin(); out_it != output_colorLabels.end() && outExist; out_it++)
        // {
        //     outExist = *out_it != it->label;
        // }
        if (!outExist)
            continue;
        if (!it->isInside(color))
            continue;

        // candidateColorIdxList.insert(candidateColorIdxList.end(),
        //                 it->intersectlabelIndexList.begin(), it->intersectlabelIndexList.end());
        output_colorLabels.push_back(it->label);
    }
    if (output_colorLabels.empty())
    {
        output_colorLabels.push_back(color_unknow);
    }

    return output_colorLabels;
}

std::vector<std::vector<unsigned char>> HLSColorClassifier::classify(unsigned char *array, unsigned int size, int *maxLabelCount)
{
    std::vector<std::vector<unsigned char>> result;
    if (size % 3 != 0)
        return result;

    *maxLabelCount = 0;
    int colorCount = size/3;
    result.reserve(colorCount);
    for (int i = 0; i < size; i+=3)
    {
        Color_t color(array[i], array[i + 1], array[i + 2]);
        auto labels = classify(color);
        result.emplace_back(labels);
        *maxLabelCount = (*maxLabelCount < labels.size()) ? labels.size() : *maxLabelCount;
    }

    return result;
}

std::vector<unsigned char> HLSColorClassifier::classifyStatistics(cv::Mat image, int sampleNumber, int cvtColorCode, cv::Mat mask, bool usePercent)
{
    std::vector<unsigned char> result;
    if (image.channels() != 3)
        return result;

    cv::Mat sampleImg, sampleMask;
    std::vector<float> result_f;
    int imgArea = image.size().area();
    if (mask.empty())
        mask = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(255));

    // get sample
    if (imgArea > sampleNumber)
    {
        float oneSampleArea = float(imgArea) / sampleNumber;
        float sampleEdge = sqrt(oneSampleArea);

        int sampleRowCount = ceil(image.rows / sampleEdge);
        int sampleColCount = ceil(image.cols / sampleEdge);

        int *rowIds = new int[sampleRowCount];
        int *colIds = new int[sampleColCount];

        float gap, val;
        gap = (float)image.rows / sampleRowCount;
        rowIds[0] = val = int(gap/2);
        for (int i = 1; val < image.rows && i < sampleRowCount; rowIds[i++] = int(val += gap));
        gap = (float)image.cols / sampleColCount;
        colIds[0] = val = int(gap/2);
        for (int i = 1; val < image.cols && i < sampleColCount; colIds[i++] = int(val += gap));

        sampleImg = cv::Mat(sampleRowCount, sampleColCount, CV_8UC3);
        sampleMask = cv::Mat(sampleRowCount, sampleColCount, CV_8UC1);
        for (int row = 0; row < sampleRowCount; row++)
        {
            for (int col = 0; col < sampleColCount; col++)
            {
                sampleImg.ptr<cv::Vec3b>(row,col)[0] =
                    image.ptr<cv::Vec3b>(rowIds[row], colIds[col])[0];
                sampleMask.ptr<uchar>(row,col)[0] =
                    mask.ptr<uchar>(rowIds[row], colIds[col])[0];
            }
        }

        delete rowIds;
        delete colIds;
    }
    else
    {
        sampleImg = image.clone();
        sampleMask = mask;
    }

    // classify color
    cv::cvtColor(sampleImg, sampleImg, cvtColorCode);
    result_f.assign(maxColorLabelNum + 1, 0.0f);
    cv::Vec3b *ptrImg = sampleImg.ptr<cv::Vec3b>(0,0);
    uchar *ptrMask = sampleMask.ptr<uchar>(0,0);
    int sampleImgArea = sampleImg.rows * sampleImg.cols;
    int colorCount = 0;
    for (int ptr_i = 0; ptr_i < sampleImgArea; ptr_i++)
    {
        if (*ptrMask++ <= 0)
        {
            ptrImg++;
            continue;
        }
        std::vector<unsigned char> labels = classify(Color_t((*ptrImg)[0], (*ptrImg)[1], (*ptrImg)[2]));
        float _scale = 1.0f/labels.size();

        for (int label_i = 0; label_i < labels.size(); label_i++)
        {
            result_f[int(labels[label_i])] += _scale;
        }
        ptrImg++;
        colorCount++;
    }
    result.assign(result_f.size(), 0);

    // transform to output datatype
    float scale = float(usePercent ? 100 : 255) / colorCount;
    for (int _i = 0; _i < result_f.size(); _i++)
    {
        float val = result_f[_i] * scale + 0.4;
        result[_i] = unsigned char(val > 255 ? 255 : val);
    }

    return result;
}


void HLSColorClassifier::setNewColorRange(std::vector<ColorRange> colorRangeMap)
{
    this->colorRangeMap.clear();
    this->colorRangeMap = colorRangeMap;
    checkRangeHaveIntersect();
}

void HLSColorClassifier::setDefaultColorRange()
{
    colorRangeMap.clear();
    colorRangeMap = std::vector<ColorRange>{

        ColorRange(Color_t(  0,   0,   0), Color_t(255,  30, 255), color_black ), // black
        ColorRange(Color_t(  0,  14,   0), Color_t(255, 204,  50), color_gray  ), // gray
        // ColorRange(Color_t(  0,   6,   0), Color_t(255, 204,  50), color_gray  ), // gray test
        ColorRange(Color_t(  0, 180,   0), Color_t(255, 255, 255), color_white ), // white

        ColorRange(Color_t(  0,  14,  50), Color_t(  8, 204, 255), color_red   ), // red
        ColorRange(Color_t(  8,  76,  50), Color_t( 21, 204, 255), color_orange), // orange
        ColorRange(Color_t(  8,  14,  50), Color_t( 21,  76, 255), color_brown ), // brown
        ColorRange(Color_t( 21,  14,  50), Color_t( 35, 204, 255), color_yellow), // yellow
        ColorRange(Color_t( 35,  14,  50), Color_t( 80, 204, 255), color_green ), // green
        ColorRange(Color_t( 80,  14,  50), Color_t( 95, 204, 255), color_cyan  ), // cyan
        ColorRange(Color_t( 95,  14,  50), Color_t(130, 204, 255), color_blue  ), // blue
        ColorRange(Color_t(130,  14,  50), Color_t(169, 204, 255), color_purple), // purple
        ColorRange(Color_t(169,  14,  50), Color_t(180, 204, 255), color_red   ), // red

    };

    checkRangeHaveIntersect();
}

void HLSColorClassifier::checkRangeHaveIntersect()
{
    maxColorLabelNum = 0;
    for (int i = 0; i < colorRangeMap.size(); i++)
    {
        colorRangeMap[i].intersectlabelIndexList.clear();
        maxColorLabelNum = (maxColorLabelNum < colorRangeMap[i].label) ?
                    colorRangeMap[i].label : maxColorLabelNum;
    }

    for (int i = 0; i < colorRangeMap.size(); i++)
    {
        auto *left = &colorRangeMap[i];
        for (int j = i+1; j < colorRangeMap.size(); j++)
        {
            auto *right = &colorRangeMap[j];
            if (left->label == right->label)
                continue;
            ColorRange intersect = *left & *right;
            if (!intersect.empty())
            {
                left->intersectlabelIndexList.push_back(j);
            }
        }
    }
}


