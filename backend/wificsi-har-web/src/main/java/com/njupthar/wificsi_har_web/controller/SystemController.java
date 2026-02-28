//package com.njupthar.wificsi_har_web.controller;
//
//import com.fasterxml.jackson.databind.ObjectMapper;
//import com.njupthar.wificsi_har_web.dto.ApiResponse;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.web.bind.annotation.*;
//
//import java.io.File;
//import java.util.HashMap;
//import java.util.Map;
//
//@RestController
//@RequestMapping("/api/v1/system")
//@CrossOrigin
//public class SystemController {
//
//    @Value("${matlab.projectDir}")
//    private String matlabProjectDir;
//
//    private final ObjectMapper mapper = new ObjectMapper();
//
//    /**
//     * GET /api/v1/system/status
//     */
//    @GetMapping("/status")
//    public ApiResponse<?> status() {
//        Map<String, Object> data = new HashMap<>();
//
//        try {
//            File trainSummary = new File(matlabProjectDir + File.separator + "models" + File.separator + "train_summary.json");
//            File thresholds = new File(matlabProjectDir + File.separator + "models" + File.separator + "thresholds.json");
//
//            boolean modelReady = trainSummary.exists() && thresholds.exists();
//
//            data.put("modelReady", modelReady);
//            data.put("trainSummaryExists", trainSummary.exists());
//            data.put("thresholdsExists", thresholds.exists());
//
//            if (trainSummary.exists()) {
//                Map<?, ?> json = mapper.readValue(trainSummary, Map.class);
//                data.put("trainSummary", json);
//            }
//
//            if (thresholds.exists()) {
//                Map<?, ?> json = mapper.readValue(thresholds, Map.class);
//                data.put("thresholds", json);
//            }
//
//            return ApiResponse.success(data);
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            return ApiResponse.fail("status check failed: " + e.getMessage());
//        }
//    }
//}
