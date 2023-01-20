package org.ea.mlp;

import java.io.File;
import java.util.*;

public class TestApplication {
    static Map<String, Integer> imageMap = new HashMap<String, Integer>();
    static ArrayList<String> test = new ArrayList<>();

    public static void main(String[] arg) {

        imageMap.put("images/img00.bmp", 1);
        imageMap.put("images/img01.bmp", 2);
        imageMap.put("images/img02.bmp", 7);
        imageMap.put("images/img03.bmp", 3);
        imageMap.put("images/img04.bmp", 4);
        imageMap.put("images/img05.bmp", 5);
        imageMap.put("images/img06.bmp", 6);
        imageMap.put("images/img07.bmp", 7);
        imageMap.put("images/img08.bmp", 8);
        imageMap.put("images/img09.bmp", 9);
        imageMap.put("images/img10.bmp", 4);
        imageMap.put("images/img11.bmp", 2);
        imageMap.put("images/img12.bmp", 1);
        imageMap.put("images/img13.bmp", 3);
        imageMap.put("images/img14.bmp", 5);
        imageMap.put("images/img15.bmp", 9);
        imageMap.put("images/img16.bmp", 6);
        imageMap.put("images/img17.bmp", 8);
        imageMap.put("images/img18.bmp", 7);
        imageMap.put("images/img19.bmp", 0);
        imageMap.put("images/img20.bmp", 2);
        imageMap.put("images/img21.bmp", 0);
        imageMap.put("images/img22.bmp", 3);

        List<String> keys = new ArrayList(imageMap.keySet());
        Collections.sort(keys);

        int correct = 0;

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(64, 10, 24, 4);
        if(!mlp.trainNetwork(0.04f, 0.01f, 0.4f, imageMap, 10000, 500)) {
            System.out.println("Error while training ... Quitting\n\r");
            System.exit(0);
        }

        for(String key : keys) {
            int answer = imageMap.get(key);
            ImageReader ir = new ImageReader();
            correct += mlp.recallNetwork(key, ir.readImage(key), answer);
        }
        System.out.println(correct + " / " + imageMap.size());
        System.out.println();

        //testing
        File folder = new File("test/");
        File[] listOfFiles = folder.listFiles();

        for (File file : listOfFiles) {
            if (file.isFile()) {
                test.add("test/" + file.getName());
            }
        }

        Collections.sort(test);
        for(String key : test) {
            ImageReader ir = new ImageReader();
            System.out.println(key + " result " + mlp.recallNetworkTest(ir.readImage(key)));
        }
    }
}
