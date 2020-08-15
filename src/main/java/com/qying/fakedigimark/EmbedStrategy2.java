package com.qying.fakedigimark;

import com.qying.fakedigimark.Util.ByteMatrix;
import com.qying.fakedigimark.Util.RSUtil;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

public class EmbedStrategy2 {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        encode("newStrategy.bmp","123",Settings.matrixWidth,Settings.matrixHeight);
    }

    public static void encode(String filepath,String content,int matrixWidth,int matrixHeight){

        int embedlen = Settings.embedLength;
        while(content.length()<embedlen)
            content = content+' ';

        int QUIET_ZONE_SIZE = Settings.QUIET_ZONE_SIZE;
        Mat blank =new Mat(matrixHeight+2*QUIET_ZONE_SIZE,matrixWidth+2*QUIET_ZONE_SIZE, CvType.CV_8UC3);
        int ROWS = blank.rows();int COLS = blank.cols();
        for(int i=0;i<ROWS;i++){
            for(int j=0;j<COLS;j++){
                blank.put(i,j,new double[]{255*Settings.baseColor,255*Settings.baseColor,255*Settings.baseColor});
            }
        }
        // Get Bitstream of watermark
        int numInputBytes = content.length();
        byte[] dataBytes = new byte[numInputBytes*2+2];
        dataBytes[0] = (byte)15;dataBytes[1] = (byte)15;
        for(int i=0;i<content.length();i+=2){
            byte b = (byte)content.charAt(i);
            dataBytes[i+2] = (byte)((b&0xF0)>>4);
            dataBytes[i+3] = (byte)((b&0x0F));
        }


        //信息嵌入
        int batchWidth = 4;
        int batchIndex = 0;
        boolean previousRed = false;
        int row = 1;int col = 1;

        while(row<matrixHeight/batchWidth-1) {
            while(col<matrixWidth/batchWidth-1){

                int index = batchIndex%dataBytes.length;
                Mat block = getImageValue(blank, row*batchWidth, col*batchWidth, batchWidth);

                //set the matrix's R or G channel 255
                for (int m = 0; m < batchWidth; m++) {
                    for (int t = 0; t < batchWidth; t++) {
                        double[] e = block.get(m, t);
                        if(previousRed) e[2] = 255;
                        else e[1] = 255;
                        blank.put(row*batchWidth + m, col*batchWidth + t, e);
                    }
                }

                previousRed = !previousRed;



                // 保证相邻两个小块至少距离为1
                col+=(2+dataBytes[index]);
                System.out.println("Moving "+dataBytes[index]);
                batchIndex++;

            }

             //超出一列的最右端，从下一行的最左端开始
            row++;col -= (matrixWidth/batchWidth-1);


        }

        Imgcodecs.imwrite(filepath, blank);

    }

    public static Mat getImageValue(Mat YMat, int x, int y, int length) {
        Mat mat = new Mat(length,length, CvType.CV_8UC3);
        for(int i=0;i<length;i++) {
            for(int j=0;j<length;j++) {
                double[] temp = YMat.get(x+i, y+j);
                // mat.put(i, j, 1.0);
                mat.put(i,j,temp);
            }
        }
        return mat;
    }



    public static void modifyDct32(Mat blank,int xbegin,int ybegin){
        //嵌入4个连续的block
//        int i=0,j=0;
//        for(int i=0;i<=8;i+=8) {
//            for (int j = 0; j <= 8; j += 8) {

                Mat block = ImgWatermarkUtil.getImageValue(blank, xbegin, ybegin, 32);
                //对分块进行DCT变换
                Core.dct(block, block);
                int[] x = Settings.x;int[] y = Settings.y;

                double dense = Settings.strength;
//            double[] a = block.get(x1,y1);
//            double[] c = block.get(x2,y2);
                for (int i = 0; i < 32; i++) {
                    for (int j = 0; j < 32; j++) {
                        if(i+j>32-4 && i+j<32+4 && i>j)
                            block.put(i, j, dense);
                    }

                }

                //对上面分块进行IDCT变换
                Core.idct(block, block);
                for (int m = 0; m < 32; m++) {
                    for (int t = 0; t < 32; t++) {
                        double[] e = block.get(m, t);
//                    System.out.print(e[0]+" ");
                        blank.put(xbegin + m, ybegin + t, (int) (e[0]));
//                    System.out.print("("+blank.get(locX[i] + m,locY[i] + t)[0]+") ");
                    }
//                System.out.println();
                }
//            }
//        }

    }


}

