package com.qying.fakedigimark.Util;

import com.qying.fakedigimark.reedsolomon.GenericGF;
import com.qying.fakedigimark.reedsolomon.ReedSolomonEncoder;

public class RSUtil {
    public static byte[] generateECBytes(byte[] dataBytes, int numEcBytesInBlock) {//生成纠错比特
        int numDataBytes = dataBytes.length;
        int[] toEncode = new int[numDataBytes + numEcBytesInBlock];
        for (int i = 0; i < numDataBytes; i++) {
            toEncode[i] = dataBytes[i] & 0xFF;
        }
        new ReedSolomonEncoder(GenericGF.QR_CODE_FIELD_256).encode(toEncode, numEcBytesInBlock);

        byte[] ecBytes = new byte[numEcBytesInBlock];
        for (int i = 0; i < numEcBytesInBlock; i++) {
            ecBytes[i] = (byte) toEncode[numDataBytes + i];
        }
        return ecBytes;
    }
}
