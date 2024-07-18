/**************************************************************************************************************
* An occlusion-resistant circle detector using inscribed triangles source code.
* Copyright (c) 2021, Mingyang Zhao
* E-mails of the authors: zhaomingyang16@mails.ucas.ac.cn
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.

* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

* By using this implementation, please cite the following paper:
*
* M. Zhao, X. Jia, D.Y "An occlusion-resistant circle detector using inscribed triangles,"
*     Pattern Recognition (2021).
**************************************************************************************************************/

#include "EDLib.h"
#include <iostream>
#include<opencv2/imgproc.hpp>
#include <ctime>
#include <random>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

/*---set thresholds---*/
// For better performance, you can mainly tune the parameters:  'T_inlier' and 'sharp_angle' 
typedef struct threshold {
	int T_l = 20;
	float T_ratio = 0.001;
	int T_o = 5;// 5 10 15 20 25
	int T_r = 5;// 5 10 15 20 25
	float T_inlier = 0.35;//0.3 0.35 0.4 0.45 0.5 (the larger the more strict)
	float T_angle = 2.0;// 
	float T_inlier_closed = 0.5;//0.5,0.6 0.7 0.8,0.9
	float sharp_angle = 60;//35 40 45 50 55 60 

}T;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>\n";
        return 1;
    }

    fs::path input_path = argv[1];
    fs::path output_path = argv[2];

    if (!fs::exists(input_path) || !fs::is_directory(input_path)) {
        std::cerr << "Error: Input directory " << input_path << " does not exist or is not a directory.\n";
        return 1;
    }

    if (!fs::exists(output_path)) {
        if (!fs::create_directory(output_path)) {
            std::cerr << "Error: Could not create output directory " << output_path << ".\n";
            return 1;
        }
    } else if (!fs::is_directory(output_path)) {
        std::cerr << "Error: Output path " << output_path << " is not a directory.\n";
        return 1;
    }

    T test_threshold;

    vector<cv::String> Filenames;
    cv::glob(input_path.string(), Filenames);
    float fmeasureSum = 0.0;
    float precisionSum = 0.0;
    float recallSum = 0.0;
    float timeSum = 0.0;

    for (const auto& file : Filenames) {
        cv::String::size_type pos1, pos2;
        pos1 = file.find("1");
        pos2 = file.find(".");
        cv::String prefix = file.substr(pos1 + 2, pos2 - pos1 - 2);
        cv::String suffix = file.substr(pos2 + 1, pos2 + 3);
        cv::String saveName = (output_path / (prefix + "_det." + suffix)).string();

        Mat testImgOrigin = imread(file, 1);
        Mat testImg = testImgOrigin.clone();
        cvtColor(testImg, testImg, COLOR_BGR2GRAY);
        GaussianBlur(testImg, testImg, Size(9, 9), 2, 2);

        // Here, the window creation and display is commented out to avoid interruption.
        // You can uncomment these lines for debugging purposes.
        // cv::imshow("Clone Image", testImg);
        // cv::waitKey();

        int height = testImg.rows;
        int width = testImg.cols;

        clock_t start, finish;
        start = clock();

        EDPF testEDPF = EDPF(testImg);
        Mat edgePFImage = testEDPF.getEdgeImage();
        Mat edge = edgePFImage.clone();
        edge = edge * -1 + 255;

        // Displaying the edge image
        // cv::imshow("Edge Image Parameter Free", edge);
        // cv::waitKey();

        vector<vector<Point>> EDPFsegments = testEDPF.getSegments();

        Mat test10 = Mat(height, width, CV_8UC1, Scalar(255));
        cvtColor(test10, test10, COLOR_GRAY2BGR);
        for (const auto& segment : EDPFsegments) {
            Scalar SegEdgesColor(rand() % 256, rand() % 256, rand() % 256);
            for (size_t i = 0; i < segment.size() - 1; ++i) {
                line(test10, segment[i], segment[i + 1], SegEdgesColor, 2);
            }
        }
        // cv::imshow("Edge Segments image", test10);
        // cv::waitKey();

        vector<vector<Point>> edgeList;
        for (const auto& segment : EDPFsegments) {
            if (segment.size() >= 16) {
                edgeList.push_back(segment);
            }
        }

        auto closedAndNotClosedEdges = extractClosedEdges(edgeList);
        vector<vector<Point>> closedEdgeList = closedAndNotClosedEdges->closedEdges;

        vector<vector<Point>> segList;
        for (const auto& edge : edgeList) {
            vector<Point> segTemp;
            RamerDouglasPeucker(edge, 2.5, segTemp);
            segList.push_back(segTemp);
        }

        auto newSegEdgeList = rejectSharpTurn(edgeList, segList, test_threshold.sharp_angle);
        vector<vector<Point>> newSegList = newSegEdgeList->new_segList;
        vector<vector<Point>> newEdgeList = newSegEdgeList->new_edgeList;

        Mat test2 = Mat(height, width, CV_8UC1, Scalar(255));
        cvtColor(test2, test2, COLOR_GRAY2BGR);
        for (size_t j = 0; j < newSegList.size(); ++j) {
            Scalar colorSharpTurn(rand() % 256, rand() % 256, rand() % 256);
            for (size_t k = 0; k < newEdgeList[j].size() - 1; ++k) {
                line(test2, newEdgeList[j][k], newEdgeList[j][k + 1], colorSharpTurn, 2);
            }
        }
        // cv::imshow("After sharp turn", test2);
        // cv::waitKey();

        auto newSegEdgeListAfterInflexion = detectInflexPt(newEdgeList, newSegList);
        vector<vector<Point>> newSegListAfterInflexion = newSegEdgeListAfterInflexion->new_segList;
        vector<vector<Point>> newEdgeListAfterInfexion = newSegEdgeListAfterInflexion->new_edgeList;

        auto it = newEdgeListAfterInfexion.begin();
        while (it != newEdgeListAfterInfexion.end()) {
            Point edgeSt = it->front();
            Point edgeEd = it->back();
            int midIndex = it->size() / 2;
            Point edgeMid = (*it)[midIndex];

            double distStEd = norm(edgeSt - edgeEd);
            double distStMid = norm(edgeSt - edgeMid);
            double distMidEd = norm(edgeMid - edgeEd);
            double distDifference = abs((distStMid + distMidEd) - distStEd);

            if (it->size() <= test_threshold.T_l || distDifference <= test_threshold.T_ratio * (distStMid + distMidEd)) {
                it = newEdgeListAfterInfexion.erase(it);
            } else {
                ++it;
            }
        }

        Mat test11 = Mat(height, width, CV_8UC1, Scalar(255));
        cvtColor(test11, test11, COLOR_GRAY2BGR);
        for (const auto& edge : newEdgeListAfterInfexion) {
            Scalar colorAfterDeleteLinePt(rand() % 256, rand() % 256, rand() % 256);
            for (size_t j = 0; j < edge.size() - 1; ++j) {
                line(test11, edge[j], edge[j + 1], colorAfterDeleteLinePt, 2);
            }
        }
        // cv::imshow("After short and line segments remove", test11);
        // cv::waitKey();

        auto closedAndNotClosedEdges1 = extractClosedEdges(newEdgeListAfterInfexion);
        vector<vector<Point>> closedEdgeList1 = closedAndNotClosedEdges1->closedEdges;
        vector<vector<Point>> notclosedEdgeList1 = closedAndNotClosedEdges1->notClosedEdges;

        Mat test4 = Mat(height, width, CV_8UC1, Scalar(255));
        cvtColor(test4, test4, COLOR_GRAY2BGR);
        for (const auto& edge : closedEdgeList1) {
            Scalar colorClosedEdges(rand() % 256, rand() % 256, rand() % 256);
            for (size_t j = 0; j < edge.size() - 1; ++j) {
                line(test4, edge[j], edge[j + 1], colorClosedEdges, 2);
            }
        }
        // cv::imshow("closedEdges2", test4);
        // cv::waitKey();

        auto sortedEdgeList = sortEdgeList(notclosedEdgeList1);

        auto arcs = coCircleGroupArcs(sortedEdgeList, test_threshold.T_o, test_threshold.T_r);
        vector<vector<Point>> groupedArcs = arcs->arcsFromSameCircles;
        vector<vector<Point>> groupedArcsThreePt = arcs->arcsStartMidEnd;
        vector<Vec3f> groupedOR = arcs->recordOR;

        vector<Circles> groupedCircles = circleEstimateGroupedArcs(groupedArcs, groupedOR, groupedArcsThreePt, test_threshold.T_inlier, test_threshold.T_angle);

        for (const auto& edge : closedEdgeList1) {
            closedEdgeList.push_back(edge);
        }

        vector<Circles> closedCircles = circleEstimateClosedArcs(closedEdgeList, test_threshold.T_inlier_closed);

        vector<Circles> totalCircles;
        if (!groupedCircles.empty()) {
            totalCircles = groupedCircles;
        }
        totalCircles.insert(totalCircles.end(), closedCircles.begin(), closedCircles.end());

        finish = clock();
        vector<Circles> preCircles = clusterCircles(totalCircles);
        timeSum += ((float)(finish - start) / CLOCKS_PER_SEC);

        Mat detectCircles = drawResult(true, testImgOrigin, saveName, preCircles);
    }

    float avePre = precisionSum / Filenames.size();
    float aveRec = recallSum / Filenames.size();
    float aveTime = timeSum / Filenames.size();
    float aveFmea = 2 * avePre * aveRec / (avePre + aveRec);
    cout << "Pre Rec Fmea Time: " << avePre << " " << aveRec << " " << aveFmea << " " << aveTime << endl;
    // waitKey(0);

    return 0;
}