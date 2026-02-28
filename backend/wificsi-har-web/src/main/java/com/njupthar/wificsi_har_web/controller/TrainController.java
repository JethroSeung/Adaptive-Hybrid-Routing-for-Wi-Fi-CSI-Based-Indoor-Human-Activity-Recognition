package com.njupthar.wificsi_har_web.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.njupthar.wificsi_har_web.dto.ApiResponse;
import com.njupthar.wificsi_har_web.dto.TrainRequest;
import com.njupthar.wificsi_har_web.service.MatlabService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/train")
@CrossOrigin
public class TrainController {

    private final MatlabService matlabService;
    private final ObjectMapper objectMapper;

    public TrainController(MatlabService matlabService) {
        this.matlabService = matlabService;
        this.objectMapper = new ObjectMapper();
    }

    @PostMapping
    public ApiResponse<?> train(@RequestBody TrainRequest req) {

        // 1️⃣ 参数校验
        if (req == null || req.getDatasetPath() == null
                || req.getDatasetPath().trim().isEmpty()) {
            return ApiResponse.fail("datasetPath cannot be empty");
        }

        try {
            // 2️⃣ 调用 Matlab
            String resultJson =
                    matlabService.trainModel(req.getDatasetPath().trim());

            // 3️⃣ 将 JSON 字符串转换成对象
            Object resultObject =
                    objectMapper.readValue(resultJson, Object.class);

            // 4️⃣ 返回结构化数据
            return ApiResponse.success("training finished", resultObject);

        } catch (Exception e) {
            e.printStackTrace();
            return ApiResponse.fail("training failed: " + e.getMessage());
        }
    }
}