package com.example.patientreadmission.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Controller
@RequestMapping("/patient")
public class PatientController {

    @Autowired
    private RestTemplate restTemplate;

    @GetMapping("/form")
    public String showForm() {
        return "patient_form"; // patient_form.html inside templates
    }

    @PostMapping("/predict")
    public String predictRisk(@RequestParam int age,
                               @RequestParam int num_procedures,
                               @RequestParam int num_medications,
                               @RequestParam int time_in_hospital,
                               Model model) {

        String url = "http://localhost:5001/predict";

        Map<String, Object> input = new HashMap<>();
        input.put("age", age);
        input.put("num_procedures", num_procedures);
        input.put("num_medications", num_medications);
        input.put("time_in_hospital", time_in_hospital);

        try {
            String rawResponse = restTemplate.postForObject(url, input, String.class);

            // Parse JSON properly
            ObjectMapper mapper = new ObjectMapper();
            Map<String, String> responseMap = mapper.readValue(rawResponse, Map.class);
            String prediction = responseMap.get("prediction");

            model.addAttribute("prediction", prediction);
        } catch (Exception e) {
            model.addAttribute("prediction", "Error contacting Flask API!");
        }

        return "prediction_result"; // returns prediction_result.html
    }
}
