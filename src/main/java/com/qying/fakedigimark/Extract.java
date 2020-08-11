package com.qying.fakedigimark;

import com.qying.fakedigimark.Util.ImageUtil;
import com.qying.fakedigimark.Util.ResultPoint;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

public class Extract {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        extract("res_attack2.bmp");
    }

    public static void extract(String filename){
        Mat blank = Imgcodecs.imread(filename,0);
        TreeMap<Double,int[]> map = new TreeMap<>();
        for(int i=0;i<blank.rows();i+=8){
            for(int j=0;j<blank.cols();j+=8) {
                queryDCT(blank,i,j,map);

            }
        }
        // find largest three points, each should keep distance with others
        ResultPoint[] keyPoints = new ResultPoint[3];int found = 0;
        outer:while(map.size()!=0){
            double key = map.lastKey();int[] candidate = map.get(key);
            System.out.println(key+" :["+candidate[0]+","+candidate[1]+"]");
            ResultPoint point = new ResultPoint(candidate[0],candidate[1]);
            map.remove(map.lastKey());
            // check distance
            boolean valid = true;
            for(int i=0;i<found;i++){
                ResultPoint existPoint = keyPoints[i];
                double distance = ResultPoint.distance(existPoint,point);
                if(distance<Settings.safeDistance){
                    valid = false;break;
                }
            }
            if(valid){
                keyPoints[found] = point;
                found++;
                if(found==3){
                    System.out.println("[Found 3 key points successfully!]");
                    break outer;
                }
            }
        }
        //conduct warpPerspective
        ResultPoint.orderBestPatterns(keyPoints);
        List<Point> listDsts = new LinkedList<>();
        listDsts.add(new Point(0,Settings.matrixHeight));
        listDsts.add(new Point(0,0));
        listDsts.add(new Point(Settings.matrixWidth,0));
        listDsts.add(new Point(Settings.matrixWidth,Settings.matrixHeight));
        List<Point> listSrcs = new LinkedList<>();
        // 得到第4个点
        Point fourth = new Point(keyPoints[2].getX()+keyPoints[0].getX()-keyPoints[1].getX(),
                keyPoints[2].getY()+keyPoints[0].getY()-keyPoints[1].getY());
        for(ResultPoint r:keyPoints)
            listSrcs.add(new Point(r.getX(),r.getY()));
        listSrcs.add(fourth);
//        blank = ImageUtil.warpPerspective(blank,listSrcs,listDsts);

        int matrixWidth = blank.rows();

        //get Watermark
//        StringBuilder str = new StringBuilder();StringBuilder tmp = new StringBuilder();

        Random r=new Random(Settings.seed);//r.nextInt()
        int embedlen = Settings.embedLength;int embedtime = Settings.embedTimes;int QUIET_ZONE_SIZE = Settings.QUIET_ZONE_SIZE;
        int batchWidth = Settings.batchWidth;int batchNum = matrixWidth/batchWidth;
        int[] extractArray = new int[embedlen*embedtime*8];
        Set<Integer> used = new HashSet<>();
        for(int times = 0;times<embedtime;times++) {
            for (int i = 0; i < embedlen; i++) {
                for (int j = 0; j < 8; j++) {
                    int index = Math.abs(r.nextInt()) % (batchNum * batchNum);

                    int row = index / batchNum;
                    int col = index % batchNum;
                    if ((row == 0 || row == batchNum - 1) || (col == 0 || col == batchNum - 1)) {
                        System.out.println("检测区域不可用，Net randInt");
                        j--;
                    } else if (used.contains(row * batchNum + col)) {
                        System.out.println("这个区域已经用过，Net randInt");
                        j--;
                    } else {
                        double k = 0.0;
                        for(int rowBias=0;rowBias<=8;rowBias+=8){
                            for(int colBias=0;colBias<=8;colBias+=8) {
                                Mat block = ImgWatermarkUtil.getImageValue(blank, QUIET_ZONE_SIZE + row * batchWidth,
                                        QUIET_ZONE_SIZE + col * batchWidth, 8);
                                Core.dct(block, block);

                                double sum1 = 0, sum2 = 0;
                                for (int z = 0; z < Settings.x.length; z++) {
                                    int ind1 = Settings.x[z], ind2 = Settings.y[z];
                                    sum1 += Math.abs(block.get(ind1, ind2)[0]);
                                    sum2 += Math.abs(block.get(ind2, ind1)[0]);
                                }
                                k+= (sum2 - sum1);// > Settings.threshold) ? 1.0 : 0.0;
                            }
                        }

                        int bit = (k>4*Settings.threshold)?1:0;
                        //int k = (sum2 - sum1 > Settings.threshold) ? 1 : 0;
                        System.out.println("Embedded in: " + row + " " + col + " " + bit);

                        extractArray[times*(Settings.embedLength*8)+i*8+(7-j)] = bit;

                        used.add(row * batchNum + col);
                        //                    if(used.size()==(batchNum-1)*(batchNum-1))
                        //                        throw new Exception("嵌入失败，容量超出");


                    }


                }

            }
            System.out.println("Time "+times+" Finish.");
        }

        //转文字
        StringBuilder str = new StringBuilder();
        for(int i=0;i<embedlen*8;i++){
            double sum = 0;
            for(int j=0;j<embedtime*(embedlen*8);j+=(embedlen*8)){
                sum+=extractArray[i+j];
            }
            extractArray[i] = (sum>embedtime/2.0)?1:0;
        }
        for(int i=0;i<embedlen;i++){
            int sum = 0;
            for(int j=0;j<8;j++){
                sum+=extractArray[8*i+j]<<(7-j);
            }
            str.append((char)sum);

        }

        System.out.println("Decoded Watermark: "+str.toString());
        System.out.println("Target File: "+filename);


    }

    public static void queryDCT(Mat blank, int xbegin, int ybegin, TreeMap<Double,int[]> map){
        //嵌入4个连续的block
//        int i=0,j=0;
        Mat block = ImgWatermarkUtil.getImageValue(blank, xbegin, ybegin, 8);
        Core.dct(block, block);
//        for (int i = 0; i < 8; i++) {
//            for (int j = 0; j < 8; j++) {
//                System.out.print(Math.round(block.get(i, j)[0]) + " ");
//
//            }
//            System.out.println();
//        }
        double sum1 = 0, sum2 = 0;
        int[] x = Settings.x;int[] y = Settings.y;
        for (int z = 0; z < x.length; z++) {
            int ind1 = x[z], ind2 = y[z];
            sum1 += Math.abs(block.get(ind1, ind2)[0]);
            sum2 += Math.abs(block.get(ind2, ind1)[0]);
        }
//        System.out.println("Sum1: " + sum1 + " ,Sum2: " + sum2);
        map.put(sum1-sum2+Math.random(),new int[]{xbegin,ybegin});

    }
}
