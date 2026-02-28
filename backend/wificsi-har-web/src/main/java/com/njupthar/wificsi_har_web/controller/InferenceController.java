package com.njupthar.wificsi_har_web.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.njupthar.wificsi_har_web.dto.ApiResponse;
import com.njupthar.wificsi_har_web.dto.InferenceRequest;
import com.njupthar.wificsi_har_web.service.MatlabService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/inference")
@CrossOrigin
public class InferenceController {

    private final MatlabService matlabService;

    public InferenceController(MatlabService matlabService) {
        this.matlabService = matlabService;
    }

    @PostMapping
    public ApiResponse<?> inference(@RequestBody InferenceRequest req) {

        if (req.getTestPath() == null || req.getTestPath().trim().isEmpty()) {
            return ApiResponse.fail("testPath cannot be empty");
        }

        try {
            String resultJson =
                    matlabService.predictFolder(req.getTestPath().trim());

            ObjectMapper mapper = new ObjectMapper();
            Object resultObject =
                    mapper.readValue(resultJson, Object.class);

            return ApiResponse.success("inference finished", resultObject);

        } catch (Exception e) {
            e.printStackTrace();
            return ApiResponse.fail("inference failed: " + e.getMessage());
        }
    }
}