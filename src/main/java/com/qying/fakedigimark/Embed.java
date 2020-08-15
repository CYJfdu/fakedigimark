package com.qying.fakedigimark;

import com.qying.fakedigimark.Util.ByteMatrix;
import com.qying.fakedigimark.Util.RSUtil;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

public class Embed {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        encode("res.bmp","123",Settings.matrixWidth,Settings.matrixHeight);
    }

    public static void encode(String filepath,String content,int matrixWidth,int matrixHeight){

        int embedlen = Settings.embedLength;
        while(content.length()<embedlen)
            content = content+' ';

        int QUIET_ZONE_SIZE = Settings.QUIET_ZONE_SIZE;
        Mat blank =new Mat(matrixHeight+2*QUIET_ZONE_SIZE,matrixWidth+2*QUIET_ZONE_SIZE, CvType.CV_8UC1);
        int ROWS = blank.rows();int COLS = blank.cols();
        for(int i=0;i<ROWS;i++){
            for(int j=0;j<COLS;j++){
                blank.put(i,j,255*Settings.baseColor);
            }
        }
        // Get Bitstream of watermark
        int numInputBytes = content.length();
        byte[] dataBytes = new byte[numInputBytes];
        for(int i=0;i<content.length();i++)
            dataBytes[i] = (byte)content.charAt(i);

        int[] locX = new int[]{QUIET_ZONE_SIZE,QUIET_ZONE_SIZE,ROWS-QUIET_ZONE_SIZE-16,ROWS-QUIET_ZONE_SIZE-16};
        int[] locY = new int[]{QUIET_ZONE_SIZE,COLS-QUIET_ZONE_SIZE-16,QUIET_ZONE_SIZE,COLS-QUIET_ZONE_SIZE-16};

        // 定位点体现为8*8小块DCT[4][5]系数明显大于DCT[5][4]
        for(int i=0;i<3;i++) {
            modifyDct(blank,locX[i],locY[i],true);
        }


        int[] x = Settings.x;int[] y = Settings.y;


        //尝试偏移一些，提取
//        int correct = 0;
//        for(int xbias=-4;xbias<=4;xbias++) {
//            for (int ybias = -4; ybias <= 4; ybias++) {
//
//                Mat block = ImgWatermarkUtil.getImageValue(blank, locX[0] + xbias, locY[0] + ybias, 8);
//                Core.dct(block, block);
//                for (int i = 0; i < 8; i++) {
//                    for (int j = 0; j < 8; j++) {
//                        System.out.print(Math.round(block.get(i, j)[0]) + " ");
//
//                    }
//                    System.out.println();
//                }
//                double sum1 = 0, sum2 = 0;
//                for (int z = 0; z < x.length; z++) {
//                    int ind1 = x[z], ind2 = y[z];
//                    sum1 += Math.abs(block.get(ind1, ind2)[0]);
//                    sum2 += Math.abs(block.get(ind2, ind1)[0]);
//                }
//                System.out.println("Sum1: " + sum1 + " ,Sum2: " + sum2);
//                if(sum1-sum2>Settings.threshold)
//                    correct++;
//            }
//        }
//        System.out.println("Correct: " + correct );//81次中对77次

        //信息嵌入
        int batchWidth = Settings.batchWidth;int batchNum = matrixWidth/batchWidth;
        Random r=new Random(Settings.seed);//r.nextInt()
        Set<Integer> used = new HashSet<>();
        byte[] ecBytes = new byte[0];

        // Step 7: Choose the mask pattern and set to "qrCode".系统默认为根据内容择优选择掩码
        StringBuilder str = new StringBuilder();

        for(int times = 0;times<Settings.embedTimes;times++) {
            for(int i=0;i<dataBytes.length+ecBytes.length;i++){
                byte b = (i<dataBytes.length)?dataBytes[i]:ecBytes[i-dataBytes.length];


                    for (int j = 0; j < 8; j++) {
                        int index = Math.abs(r.nextInt()) % (batchNum * batchNum);
                        int k = (b >> j) % 2;//从低位开始
                        str.append((char) ('0' + k));
                        //                if(k==1) {
                        int row = index / batchNum;
                        int col = index % batchNum;
                        if ((row == 0 || row == batchNum - 1) || (col == 0 || col == batchNum - 1)) {
                            System.out.println("检测区域不可用，Net randInt");
                            j--;
                        } else if (used.contains(row * batchNum + col)) {
                            System.out.println("这个区域已经用过，Net randInt");
                            j--;
                        } else {
                            //在选定的batch中嵌入2~4个点，规定提取出至少5个才能认为正确,是1
                            if (k == 1) {
                                //嵌入信息
                                modifyDct(blank, QUIET_ZONE_SIZE + row * batchWidth, QUIET_ZONE_SIZE + col * batchWidth, false);

                                //                        System.out.println("Embedded in: " + row + " " + k);
                            }
                            System.out.println("Embedded in: " + row + " " + col + " " + k);

                            used.add(row * batchNum + col);
                            //                    if(used.size()==(batchNum-1)*(batchNum-1))
                            //                        throw new Exception("嵌入失败，容量超出");


                        }

                        //                }
                    }

                }

            System.out.println("Time "+times+" Finish.");
        }

        Imgcodecs.imwrite(filepath, blank);

    }

    public static void modifyDct(Mat blank,int xbegin,int ybegin,boolean isRectPoint){
        //嵌入4个连续的block
//        int i=0,j=0;
        for(int i=0;i<=8;i+=8) {
            for (int j = 0; j <= 8; j += 8) {

                Mat block = ImgWatermarkUtil.getImageValue(blank, xbegin+i, ybegin+j, 8);
                //对分块进行DCT变换
                Core.dct(block, block);
                int[] x = Settings.x;int[] y = Settings.y;

                double dense = Settings.strength;
//            double[] a = block.get(x1,y1);
//            double[] c = block.get(x2,y2);
                for (int z = 0; z < x.length; z++) {
                    int ind1 = (isRectPoint) ? x[z] : y[z], ind2 = (isRectPoint) ? y[z] : x[z];
                    block.put(ind1, ind2, dense);
                    block.put(ind2, ind1, 0);

                }

                //对上面分块进行IDCT变换
                Core.idct(block, block);
                for (int m = 0; m < 8; m++) {
                    for (int t = 0; t < 8; t++) {
                        double[] e = block.get(m, t);
//                    System.out.print(e[0]+" ");
                        blank.put(xbegin+i + m, ybegin+j + t, (int) (e[0]));
//                    System.out.print("("+blank.get(locX[i] + m,locY[i] + t)[0]+") ");
                    }
//                System.out.println();
                }
            }
        }

    }


}

