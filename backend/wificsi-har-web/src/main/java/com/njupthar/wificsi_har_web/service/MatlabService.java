package com.njupthar.wificsi_har_web.service;

import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;

@Service
public class MatlabService {

    // 修改为你自己的路径
    private final String matlabExe =
            "D:\\MATLAB2024B\\bin\\matlab.exe";

    private final String matlabWorkDir =
            "C:\\Users\\LENOVO\\Desktop\\代码\\新方向\\自适应阈值 - 副本";

    private final String outputDir =
            "C:\\Users\\LENOVO\\Desktop\\WiFi-CSI-HAR-System\\matlab\\outputs";

    public String trainModel(String dataPath) throws Exception {

        String command = "train_model('" + dataPath.replace("\\", "/") + "')";

        ProcessBuilder pb = new ProcessBuilder(
                matlabExe,
                "-batch",
                command
        );

        pb.directory(new File(matlabWorkDir));
        pb.redirectErrorStream(true);

        Process process = pb.start();

        BufferedReader reader =
                new BufferedReader(new InputStreamReader(process.getInputStream()));

        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException("Matlab train failed.");
        }

        return readJson("train_result.json");
    }

    public String predictFolder(String dataPath) throws Exception {

        String command = "predict_folder('" + dataPath.replace("\\", "/") + "')";

        ProcessBuilder pb = new ProcessBuilder(
                matlabExe,
                "-batch",
                command
        );

        pb.directory(new File(matlabWorkDir));
        pb.redirectErrorStream(true);

        Process process = pb.start();

        BufferedReader reader =
                new BufferedReader(new InputStreamReader(process.getInputStream()));

        String line;
        while ((line = reader.readLine()) != null) {
            System.out.println(line);
        }

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException("Matlab predict failed.");
        }

        return readJson("predict_result.json");
    }

    private String readJson(String fileName) throws Exception {
        String path = outputDir + File.separator + fileName;
        return new String(Files.readAllBytes(Paths.get(path)));
    }
}