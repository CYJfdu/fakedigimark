package com.qying.fakedigimark.Util;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.*;

public class ImageUtil {
    /**
     * canny算法，边缘检测
     *
     * @param src
     * @return
     */
    public static Mat canny(Mat src) {
        Mat mat = src.clone();
        Imgproc.Canny(src, mat, 60, 200);
//        HandleImgUtils.saveImg(mat, "C:/Users/admin/Desktop/opencv/open/x/canny.jpg");
        return mat;
    }


    /**
     * 透视变换，矫正图像 思路： 1、寻找图像的四个顶点的坐标(重要) 思路： 1、canny描边 2、寻找最大轮廓
     * 3、对最大轮廓点集合逼近，得到轮廓的大致点集合 4、把点击划分到四个区域中，即左上，右上，左下，右下 5、根据矩形中，对角线最长，找到矩形的四个顶点坐标
     * 2、根据输入和输出点获得图像透视变换的矩阵 3、透视变换
     *
     * @param src
     */
    public static Mat warpPerspective(Mat src,List<Point> listSrcs,List<Point> listDsts) {
//        // 灰度话
//        src = HandleImgUtils.gray(src);
//        // 找到四个点
//        Point[] points = HandleImgUtils.findFourPoint(src);
//
//        // Canny
//        Mat cannyMat = HandleImgUtils.canny(src);
//        // 寻找最大矩形
//        RotatedRect rect = HandleImgUtils.findMaxRect(cannyMat);

        // 点的顺序[左上 ，右上 ，右下 ，左下]
//        List<Point> listSrcs = java.util.Arrays.asList(points[0], points[1], points[2], points[3]);
        Mat srcPoints = Converters.vector_Point_to_Mat(listSrcs, CvType.CV_32F);

//        Rect r = rect.boundingRect();
//        r.x = Math.abs(r.x);
//        r.y = Math.abs(r.y);
//        List<Point> listDsts = java.util.Arrays.asList(new Point(r.x, r.y), new Point(r.x + r.width, r.y),
//                new Point(r.x + r.width, r.y + r.height), new Point(r.x, r.y + r.height));
//
//        System.out.println(r.x + "," + r.y);

        Mat dstPoints = Converters.vector_Point_to_Mat(listDsts, CvType.CV_32F);

        Mat perspectiveMmat = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);

        Mat dst = new Mat();

        Imgproc.warpPerspective(src, dst, perspectiveMmat, src.size(), Imgproc.INTER_LINEAR + Imgproc.WARP_INVERSE_MAP,
                1, new Scalar(0));

        return dst;

    }

}
