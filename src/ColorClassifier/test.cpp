#include "ColorClassifier.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>


std::vector<cv::Vec3b> ColorLabelBGR = {
    cv::Vec3b(  0,  0,  0), cv::Vec3b(  0,  0,  0), cv::Vec3b(120,120,120), cv::Vec3b(255,255,255),  // color_unknow, color_black, color_gray, color_white,   
    cv::Vec3b(  0,  0,255), cv::Vec3b(  0,120,255), cv::Vec3b(  0,60,125),  // color_red, color_orange, color_brown, 
    cv::Vec3b(  0,255,255), cv::Vec3b(  0,255,  0), cv::Vec3b(200,200,  0),  // color_yellow, color_green, color_cyan, 
    cv::Vec3b(255,  0,  0), cv::Vec3b(200,  0,200),                // color_blue, color_purple
};

void testAllHLSColorClassify(int H_count, int L_count, int S_count)
{
    HLSColorClassifier colorClassfer;
    colorClassfer.setDefaultColorRange();

    int colorGap[3] = {H_count,L_count,S_count};
    float colorGapVal[3];
    colorGapVal[0] = 180.0f/ (float)colorGap[0];
    colorGapVal[1] = 255.0f/ (float)colorGap[1];
    colorGapVal[2] = 255.0f/ (float)colorGap[2];
    int edgeNum = sqrt(colorGap[0] * colorGap[1] * colorGap[2]);
    std::vector<cv::Mat> imgs;
    std::vector<cv::Mat> imgsRes;

    // create color map
    for (int x = 0; x < colorGap[0]; x++)
    {
        cv::Mat img(colorGap[1], colorGap[2], CV_8UC3);
        for (int y = 0; y < colorGap[1]; y++)
        {
            for (int z = 0; z < colorGap[2]; z++)
            {
                // printf("x,y,z: %d %d %d\n", x,y,z);
                img.ptr<cv::Vec3b>(y,z)[0] = cv::Vec3b(
                        int(x * colorGapVal[0]), int(y * colorGapVal[1]), int(z * colorGapVal[2]));
            }
        }
        cv::cvtColor(img, img, cv::COLOR_HLS2BGR);
        imgs.push_back(img);
    }
    
    // classify color map 
    for (int i = 0; i < imgs.size(); i++)
    {
        cv::Mat img;
        img = imgs.at(i).clone();
        cv::cvtColor(img, img, cv::COLOR_BGR2HLS);

        unsigned char *img_ptr = img.ptr<unsigned char>(0);
        std::vector<std::vector<unsigned char>> labels;
        labels = colorClassfer.classify(img_ptr, img.rows * img.cols * 3);

        std::vector<cv::Mat> resArr;
        for (int j = 0; j < 3; j++)
        {
            cv::Mat res(img.rows, img.cols, CV_8UC3, ColorLabelBGR[0]);
            cv::Vec3b *res_ptr = res.ptr<cv::Vec3b>(0);

            for (int i_label = 0; i_label < labels.size(); i_label++)
            {
                unsigned char firstLabelFlag = (j >= labels[i_label].size()) ?
                        labels[i_label][labels[i_label].size()-1] : labels[i_label][j];
                res_ptr[i_label] = ColorLabelBGR[firstLabelFlag];
            }
            resArr.push_back(res);
        }
        cv::Mat resMap;
        cv::vconcat(resArr, resMap);
        imgsRes.push_back(resMap);
    }

    for (int i = 0; i < imgs.size(); i++)
    {
        cv::vconcat(std::vector<cv::Mat>{imgs[i], imgsRes[i]}, imgs[i]);
    }
    cv::Mat show;
    cv::hconcat(imgs, show);

    cv::resize(show, show, cv::Size(show.cols*5, show.rows*5), 0.0, 0.0, cv::INTER_NEAREST);

    cv::imwrite("HLSClassifyResult.jpg", show);

}


int main()
{
    // testAllHLSColorClassify(30, 100, 100);
    // return 0;

    int imageRows = 100;
    int imageCols = 100;
    int sampleNumber = 10;
    float oneSampleArea = float(imageRows * imageCols) / sampleNumber;
    float sampleEdge = sqrt(oneSampleArea);

    int sampleRowCount = ceil(imageRows / sampleEdge);
    int sampleColCount = ceil(imageCols / sampleEdge);

    int *rowIds = new int[sampleRowCount];
    int *colIds = new int[sampleColCount];

    float gap, val;
    gap = (float)imageRows / sampleRowCount;
    rowIds[0] = val = int(gap/2);
    for (int i = 1; val < imageRows && i < sampleRowCount; rowIds[i++] = int(val += gap));

    printf("rowIds:");
    for (int i = 0; i < sampleRowCount; i++)
    {
        printf(" %d", rowIds[i]);
    }
    printf("\n");


return 0;



    HLSColorClassifier colorClassfer;

    colorClassfer.setDefaultColorRange();




    unsigned char Hue_i[11] = {0, 15, 27, 42, 73, 87, 102, 123, 137, 158, 179};
    unsigned char Saturation_i[6] = {0,30,70,125,180,235};
    unsigned char Value_i[4] =  {35, 95, 150, 210};


    
    std::vector<unsigned char> labels_2 = colorClassfer.classify(Color_t(15,0,35));
    
    for (auto &d : labels_2)
        printf(" %d", d);

#if 0
    printf("     H:");
    for (int h = 0; h < sizeof(Hue_i); h++)
    {
        printf(" %3d |", Hue_i[h]);
    }
    printf("\n");


    for (int v = 0; v < sizeof(Value_i); v++)
    {
        printf("V[%3d]:\n", Value_i[v]);
        for (int s = 0; s < sizeof(Saturation_i); s++)
        {
            printf("S[%3d]:", Saturation_i[s]);
            for (int h = 0; h < sizeof(Hue_i); h++)
            {
                Color_t color(Hue_i[h], Saturation_i[s], Value_i[v]);
                std::vector<unsigned char> labels = colorClassfer.classify(color);
                
                if (labels.size() == 0)
                {
                    printf("    ");
                }
                else
                {
                    for (int i = 0; i < labels.size(); i++)
                    {
                        printf(" %3d", labels[i]);
                    }
                }
                printf(" |");

            }
            printf("\n");
        }
        printf("\n\n");
    }
#endif

    return 0;
}