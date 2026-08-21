// include header files for STM32 HAL, BMI270 sensor, and TensorFlow Lite Micro
#include "stm32l4xx_hal.h"
#include "board_setup.h"
#include "BMI270.h"
#include "activity_model_int8.h"     //edit for updated models
#include "math.h"

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// defining activity labels
const char* activity_labels_cpp[] = {
    "Standing",
    "Walking",
    "Walking Fast",
    "Jogging"
};

// Declare I2C Handle globally
I2C_HandleTypeDef hi2c1;
// Instantiate BMI270 class globally
BMI270 bmi270_sensor(&hi2c1);
// Pointers and objects used by TFLite Micro
const tflite::Model* model = nullptr;               //initialized to nullptr to avoid dangling pointer issues
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<15> op_resolver;     //allows up to 15 operators to be registered
TfLiteTensor* input_tensor = nullptr;      
TfLiteTensor* output_tensor = nullptr;

// memory allocation  arena for network tensor processing layers
const int kTensorArenaSize = 20 * 1024;               // 20 KB
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize]; // aligned to 16 bytes for hardware vector speedup

// error handler function for I2C initialization failures
void Error_Handler() {
    UART_printf("I2C Initialization Error! Check connections and configurations.\r\n");
    while (1) {
        // Stay here if detected
    }
}

// function to initialize the I2C1 peripheral structure
void init_I2C_BMI270() {
    hi2c1.Instance = I2C1;      // use I2C1 peripheral
    // timing configuration for 100kHz Standard Mode (Assuming 80MHz internal clock)
    hi2c1.Init.Timing = 0x00702991;     // 400KHz Fast Mode timing for 80MHz clock
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.OwnAddress2 = 0;
    hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    // This call automatically triggers HAL_I2C_MspInit()
    if (HAL_I2C_Init(&hi2c1) != HAL_OK) {
        Error_Handler();
    }
    // Enable the Analog Noise Filter
    if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK) {
        Error_Handler();
    }
    // Disable the Digital Noise Filter
    if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK) {
        Error_Handler();
    }
    UART_printf("I2C1 Peripheral and Pins Initialized.\r\n");
}

// Configures low-level clocks and physical pin hardware.
void HAL_I2C_MspInit(I2C_HandleTypeDef* hi2c) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    if (hi2c->Instance == I2C1) {
        __HAL_RCC_I2C1_CLK_ENABLE();
        __HAL_RCC_GPIOB_CLK_ENABLE();
        // Change from PIN_6 | PIN_7 to PIN_8 | PIN_9
        GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_9;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
        GPIO_InitStruct.Pull = GPIO_PULLUP;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
    }
}

// Function to read IMU data from BMI270
void read_imu_data(float* input_data) {
    bmi270_sensor_data_t sensor_data;
    if (bmi270_sensor.readSensorData(&sensor_data) == 0) {
        input_data[0] = sensor_data.acc_x; // ax
        input_data[1] = sensor_data.acc_y; // ay
        input_data[2] = sensor_data.acc_z; // az
        input_data[3] = sensor_data.gyr_x;  // gx
        input_data[4] = sensor_data.gyr_y;  // gy
        input_data[5] = sensor_data.gyr_z;  // gz
    } else {
        UART_printf("Failed to read IMU data!\r\n");
    }
}

// Configures board, tflite, and sensor
void setup() {
    // 1. Hardware & Peripheral Initialization
    tflite::InitializeTarget();
    init_GPIO_pins();
    init_UART2();
    init_TIM2(); 
    HAL_Delay(100);     

    UART_printf("Initializing I2C for BMI270...\r\n");
    init_I2C_BMI270();

    UART_printf("Initializing BMI270 sensor...\r\n");
    if (bmi270_sensor.init() != 0) {
        UART_printf("Failed to initialize BMI270 sensor!\r\n");
        while(1){}      
    } else {
        UART_printf("BMI270 sensor initialized successfully!\r\n");
    }

    // 2. Load Model & Schema Check
    UART_printf("Initializing TFLite Micro...\r\n");
    model = tflite::GetModel(activity_model_int8);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        UART_printf("Model schema version mismatch! Expected %d, got %d.\r\n",
                    TFLITE_SCHEMA_VERSION, (int)model->version());
        while(1){}
    }

    // 3. Register Operators
    static tflite::MicroMutableOpResolver<20> op_resolver;
    op_resolver.AddConv2D();
    op_resolver.AddMaxPool2D();
    op_resolver.AddMean();
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax();
    op_resolver.AddLogistic();         
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();
    op_resolver.AddReshape();
    op_resolver.AddExpandDims();      
    op_resolver.AddSqueeze();         
    op_resolver.AddTranspose();       
    op_resolver.AddPad();
    op_resolver.AddMul();
    op_resolver.AddSub();
    op_resolver.AddAdd();

    // 4. Instantiate Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, g_tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    // 5. Allocate Tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        UART_printf("AllocateTensors() failed with status code: %d\r\n", (int)allocate_status);
        while (1);
    }

    // 6. Bind Input/Output Tensors
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    UART_printf("TFLite Micro initialized successfully!\r\n");
}

#define WINDOW_SIZE 80              // 2 seconds of data at 40 Hz
#define NUM_FEATURES 6              // ax, ay, az, gx, gy, gz
#define NUM_CLASSES 4               // standing, walking, walking fast, jogging
#define CONFIDENCE_THRESHOLD 0.70f  // 70% confidence threshold for predictions

static float window_buffer[WINDOW_SIZE][NUM_FEATURES];
// main inference loop: collects data, normalizes, quantizes, runs inference, and outputs results
void inference_loop() {
    const float scaler_mean[NUM_FEATURES]  = { 
        8.65645058f, -1.93121760f, -2.78044642f, -0.09216424f, 0.09234816f, -0.02100617f
    };
    const float scaler_scale[NUM_FEATURES] = { 
        4.78339646f,  2.89832250f,  3.13069982f, 36.32479483f, 57.31395915f, 36.30224995f
    };

    // 1. BLOCK & READ 80 NEW FRAMES (takes exactly 2 seconds at 40 Hz)
    UART_printf("Predicting activity...\r\n");
    for (int t = 0; t < WINDOW_SIZE; ++t) {
        float raw_imu_data[NUM_FEATURES];
        read_imu_data(raw_imu_data);

        // Normalize and store directly into time index t
        for (int f = 0; f < NUM_FEATURES; ++f) {
            window_buffer[t][f] = (raw_imu_data[f] - scaler_mean[f]) / scaler_scale[f];
        }

        HAL_Delay(25); // 40Hz sampling interval
    }

    // 2. Quantize normalized floats into TFLite INT8 input tensor
    float in_scale = input_tensor->params.scale;
    int32_t in_zero_point = input_tensor->params.zero_point;
    int8_t* input_data = input_tensor->data.int8;

    int tensor_idx = 0;
    for (int t = 0; t < WINDOW_SIZE; ++t) {
        for (int f = 0; f < NUM_FEATURES; ++f) {
            float normalized_val = window_buffer[t][f];
            
            // Quantize Float -> INT8: (val / scale) + zero_point
            int32_t quantized_val = (int32_t)(normalized_val / in_scale) + in_zero_point;
            
            // Clamp value to int8 range [-128, 127]
            if (quantized_val < -128) quantized_val = -128;
            if (quantized_val > 127)  quantized_val = 127;

            input_data[tensor_idx++] = (int8_t)quantized_val;
        }
    }

    // 3. Run Inference ONCE on the full batch
    if (interpreter->Invoke() != kTfLiteOk) {
        UART_printf("Invoke failed!\r\n\r\n");
        return;
    }

    // 4. De-quantize INT8 output scores back into probabilities
    int8_t* output_quantized = output_tensor->data.int8;
    float out_scale = output_tensor->params.scale;
    int32_t out_zero_point = output_tensor->params.zero_point;

    float max_score = -1.0f;
    int predicted_activity_index = -1;

    for (int i = 0; i < NUM_CLASSES; ++i) {
        // Convert INT8 output to float score
        float score = (output_quantized[i] - out_zero_point) * out_scale;

        if (score > max_score) {
            max_score = score;
            predicted_activity_index = i;
        }
    }

    // 5. Output display with clean percentages and double spacing (\r\n\r\n)
    if (max_score >= CONFIDENCE_THRESHOLD && predicted_activity_index != -1) {
        int confidence_pct = (int)(max_score * 100.0f);
        if (confidence_pct > 100) confidence_pct = 100; // Cap at 100%

        UART_printf("Activity: %s (Confidence: %d%%)\r\n\r\n",
                    activity_labels_cpp[predicted_activity_index],
                    confidence_pct);
    } else {
        UART_printf("Prediction uncertain (Max Score < Threshold)\r\n\r\n");
    }
}



#include <cmath>
// Functions for data collection and user interaction:

// 1. Pauses execution until you press the Blue User Button on the Nucleo board to start
void wait_for_user_ready(const char* activity_name, int activity_id) {
    // Enable GPIOC hardware clock & configure Pin 13
    __HAL_RCC_GPIOC_CLK_ENABLE();
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    UART_printf("\r\n========================================\r\n");
    UART_printf("READY FOR CLASS %d: %s\r\n", activity_id, activity_name);
    UART_printf("Adjust treadmill / get in position, then PRESS BLUE USER BUTTON...\r\n");
    UART_printf("========================================\r\n");

    // Wait until button is pressed (PIN 13 drops LOW when pressed)
    while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_SET) {
        HAL_Delay(50);
    }

    // Wait until button is released
    while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {
        HAL_Delay(50);
    }

    // 5-second countdown to get into position
    for (int i = 5; i > 0; --i) {
        UART_printf("Recording in %d...\r\n", i);
        HAL_Delay(1000);
    }
}

// 2. Function to check and manage pause/resume states during recording
void check_and_handle_pause(const char* activity_name) {
    // If the Blue User Button (PIN 13) is pressed during recording
    if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {
        UART_printf("\r\n========================================\r\n");
        UART_printf("--- RECORDING PAUSED ---\r\n");
        UART_printf("Reposition on stairs, then PRESS BLUE BUTTON to RESUME...\r\n");
        UART_printf("========================================\r\n");

        // Wait until button is released from the initial pause press
        while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {
            HAL_Delay(50);
        }

        // Wait until user presses button again to resume
        while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_SET) {
            HAL_Delay(50);
        }

        // Wait until button is released after resume press
        while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {
            HAL_Delay(50);
        }

        // 3-second countdown before logging resumes
        for (int i = 3; i > 0; --i) {
            UART_printf("Resuming %s in %d...\r\n", activity_name, i);
            HAL_Delay(1000);
        }
        UART_printf("--- RESUMED RECORDING ---\r\n\r\n");
    }
}

// 3. Collects IMU data for continuous activities (e.g., Treadmill / Flat Ground)
void collect_activity(int activity_id, const char* activity_name, uint32_t duration_ms) {
    // Wait for physical button press before starting
    wait_for_user_ready(activity_name, activity_id);

    UART_printf("========================================\r\n");
    UART_printf("========================================\r\n");
    UART_printf("--- RECORDING %s ---\r\n", activity_name);
    UART_printf("========================================\r\n");
    UART_printf("========================================\r\n");

    uint32_t start_time = HAL_GetTick();
    bmi270_sensor_data_t raw_data;

    while ((HAL_GetTick() - start_time) < duration_ms) {
        if (bmi270_sensor.readSensorData(&raw_data) == 0) {
            int ax_i = (int)raw_data.acc_x, ax_f = (int)(fabsf(raw_data.acc_x - ax_i) * 10000);
            int ay_i = (int)raw_data.acc_y, ay_f = (int)(fabsf(raw_data.acc_y - ay_i) * 10000);
            int az_i = (int)raw_data.acc_z, az_f = (int)(fabsf(raw_data.acc_z - az_i) * 10000);
            int gx_i = (int)raw_data.gyr_x, gx_f = (int)(fabsf(raw_data.gyr_x - gx_i) * 10000);
            int gy_i = (int)raw_data.gyr_y, gy_f = (int)(fabsf(raw_data.gyr_y - gy_i) * 10000);
            int gz_i = (int)raw_data.gyr_z, gz_f = (int)(fabsf(raw_data.gyr_z - gz_i) * 10000);

            UART_printf("%d.%04d,%d.%04d,%d.%04d,%d.%04d,%d.%04d,%d.%04d,%d\r\n", 
                        ax_i, ax_f, ay_i, ay_f, az_i, az_f,
                        gx_i, gx_f, gy_i, gy_f, gz_i, gz_f,
                        activity_id);
        }
        HAL_Delay(10); // 100 Hz sampling rate
    }
    UART_printf("========================================\r\n");
    UART_printf("========================================\r\n");
    UART_printf("--- COMPLETED %s ---\r\n", activity_name);
    UART_printf("========================================\r\n");
    UART_printf("========================================\r\n");
}

// 4. Collects IMU data for stairs which require mid-session pauses
void collect_activity_pausable(int activity_id, const char* activity_name, uint32_t duration_ms) {
    // Wait for physical button press before starting initial batch
    wait_for_user_ready(activity_name, activity_id);

    UART_printf("========================================\r\n");
    UART_printf("--- RECORDING %s (PAUSABLE) ---\r\n", activity_name);
    UART_printf("Note: Press BLUE BUTTON anytime to PAUSE while repositioning!\r\n");
    UART_printf("========================================\r\n");

    uint32_t active_recording_time_ms = 0;
    bmi270_sensor_data_t raw_data;

    // Accumulates ONLY active recording time so that pauses do not consume duration
    while (active_recording_time_ms < duration_ms) {
        
        // Check if user hit the button to pause execution
        check_and_handle_pause(activity_name);

        // Record data point
        if (bmi270_sensor.readSensorData(&raw_data) == 0) {
            int ax_i = (int)raw_data.acc_x, ax_f = (int)(fabsf(raw_data.acc_x - ax_i) * 10000);
            int ay_i = (int)raw_data.acc_y, ay_f = (int)(fabsf(raw_data.acc_y - ay_i) * 10000);
            int az_i = (int)raw_data.acc_z, az_f = (int)(fabsf(raw_data.acc_z - az_i) * 10000);
            int gx_i = (int)raw_data.gyr_x, gx_f = (int)(fabsf(raw_data.gyr_x - gx_i) * 10000);
            int gy_i = (int)raw_data.gyr_y, gy_f = (int)(fabsf(raw_data.gyr_y - gy_i) * 10000);
            int gz_i = (int)raw_data.gyr_z, gz_f = (int)(fabsf(raw_data.gyr_z - gz_i) * 10000);

            UART_printf("%d.%04d,%d.%04d,%d.%04d,%d.%04d,%d.%04d,%d.%04d,%d\r\n", 
                        ax_i, ax_f, ay_i, ay_f, az_i, az_f,
                        gx_i, gx_f, gy_i, gy_f, gz_i, gz_f,
                        activity_id);
        }

        HAL_Delay(10); // ~100 Hz sampling rate
        active_recording_time_ms += 10; // Increment active elapsed time
    }

    UART_printf("========================================\r\n");
    UART_printf("--- COMPLETED %s ---\r\n", activity_name);
    UART_printf("========================================\r\n");
}

// 5. Main data collectionfunction to for all 6 activity classes
void collect_all() {
    uint32_t duration_3_min = 3 * 60 * 1000; // 180,000 ms active duration per class

    UART_printf("ax,ay,az,gx,gy,gz,activity\r\n");

    // Continuous activities (Treadmill / Flat ground)
    collect_activity(1, "Standing (0 mph)", duration_3_min);
    collect_activity(2, "Walking (2 mph)", duration_3_min);
    collect_activity(3, "Fast Walking (4 mph)", duration_3_min);
    collect_activity(4, "Running (7 mph)", duration_3_min);

    // Pausable activities (Stairs)
    collect_activity_pausable(5, "Walking Upstairs", duration_3_min);
    collect_activity_pausable(6, "Walking Downstairs", duration_3_min);

    UART_printf("\r\n========================================\r\n");
    UART_printf("ALL 6 CLASSES DATA COLLECTION COMPLETE!\r\n");
    UART_printf("========================================\r\n");
}





int main(void) {
    // Resets all peripherals, initializes the flash interface and the systick.
    HAL_Init();
    // Called once to initialize the board, TensorFlow Lite Micro, and the BMI270 sensor
    setup();

        // MAIN DEAD LOOP: Uncomment the desired function to run either real-timeinference or data collection
        while (1) {
        //UNCOMMENT WHEN READY TO PERFORM INFERENCE
        inference_loop();   
        
        // copy this into terminal to start data collection:     
            //  ~/.platformio/penv/bin/pio device monitor > full_dataset.csv
        
        // Print CSV Header on startup for new data logging session
        //UART_printf("ax,ay,az,gx,gy,gz,activity\r\n");
        //UNCOMMENT WHEN READY TO COLLECT RAW IMU DATA
        //collect_all();            
            //while(1) {
                //HAL_Delay(1000);// Stay here after data collection is complete
        
            
            }
        }  
    