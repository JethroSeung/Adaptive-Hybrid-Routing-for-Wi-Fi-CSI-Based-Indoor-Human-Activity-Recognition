package com.njupthar.wificsi_har_web.dto;

import lombok.Data;

@Data
public class InferenceRequest {
    private String testPath;

    public String getTestPath() {
        return testPath;
    }

    public void setTestPath(String testPath) {
        this.testPath = testPath;
    }
}
