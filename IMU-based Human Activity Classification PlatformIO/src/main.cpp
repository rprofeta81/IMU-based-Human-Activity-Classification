// include header files for STM32 HAL, BMI270 sensor, and TensorFlow Lite Micro
#include "stm32l4xx_hal.h"
#include "board_setup.h"
#include "bmi270.h"
#include "activity_model.h"

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// defining activity labels
const char* activity_labels_cpp[] = {
    "Walking Upstairs",
    "Walking Downstairs",
    "Walking",
    "Sitting",
    "Standing",
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
const int kTensorArenaSize = 90 * 1024;               // 90 KB
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
    // target specific initialization for TensorFlow Lite Micro
    tflite::InitializeTarget();
    // initialize required pins (GPIO, UART, Timers)
    init_GPIO_pins();
    init_UART2();
    init_TIM2(); 
    HAL_Delay(100);     // Short delay to ensure UART is ready
    UART_printf("Initializing I2C for BMI270...\r\n");
    init_I2C_BMI270();

    UART_printf("Initializing BMI270 sensor...\r\n");
    // initialize the BMI270 sensor
    if (bmi270_sensor.init() != 0) {
        UART_printf("Failed to initialize BMI270 sensor! Check connections and sensor power.\r\n");
        while(1){}      //dead hang
    } else {
        UART_printf("BMI270 sensor initialized successfully!\r\n");
    }

    UART_printf("Initializing TFLite Micro...\r\n");
    model = tflite::GetModel(activity_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        UART_printf("Model schema version mismatch! Expected %d, got %d.\r\n",
                    TFLITE_SCHEMA_VERSION, (int)model->version());
        while(1){}
    }
    UART_printf("TFLite Micro initialized successfully!\r\n");

    // From TFLite include the required operators
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax();
    op_resolver.AddReshape();
    op_resolver.AddQuantize();
    op_resolver.AddDequantize();
    op_resolver.AddExpandDims(); 
    op_resolver.AddMul();
    op_resolver.AddAdd();
    op_resolver.AddSub();
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, g_tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        UART_printf("AllocateTensors() failed!\r\n");
        while(1){}
    }
    // gets pointers to the model's input and output tensors, 
    //which will be used to feed data in and read predictions out
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    UART_printf("TFLite Micro initialized successfully!\r\n");
}

//main loop that: reads IMU data, performs inference, and outputs results
void inference_loop() {
    // STM32 hardware SysTick Delay for 2.5s
    HAL_Delay(2500);   
    UART_printf("\r\nReading IMU data...\r\n");
    float raw_imu_data[6];
    read_imu_data(raw_imu_data);
    for (int i = 0; i < 6; ++i) {
        input_tensor->data.f[i] = raw_imu_data[i];
    }

    // model performs its inference on the provided input data
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        UART_printf("Invoke failed!");
        return;
    }
    
    // retrieves the prediction probabilities from the model's output tensor
    float* output_scores = output_tensor->data.f;
    // finds the index of the output_scores array that has the highest value
    float max_score = -1.0f;
    int predicted_activity_index = -1;
    for (int i = 0; i < 6; ++i) {
        if (output_scores[i] > max_score) {
            max_score = output_scores[i];
            predicted_activity_index = i;
        }
    }
    // shows results in serial monitor with the predicted activity label and its corresponding score
    if (predicted_activity_index != -1) {
        UART_printf("Predicted Activity: %s (Score: %d.%04d)\r\n",
                    activity_labels_cpp[predicted_activity_index],
                    (int)max_score,
                    (int)((max_score - (int)max_score) * 10000));
    } else {
        UART_printf("Could not determine activity.\r\n");
    }
}

// Collects raw 16-bit integer IMU readings for a specific activity ID
void collect_activity(int activity_id, const char* activity_name, uint32_t duration_ms) {
    // 10-second prep window
    UART_printf("\r\n========================================\r\n");
    UART_printf("PREPARE FOR: %s (ID: %d)\r\n", activity_name, activity_id);
    UART_printf("Set treadmill speed now!\r\n");
    UART_printf("========================================\r\n");
    
    for (int i = 10; i > 0; --i) {
        UART_printf("Starting in %d seconds...\r\n", i);
        HAL_Delay(1000);
    }

    UART_printf("--- RECORDING %s ---\r\n", activity_name);

    uint32_t start_time = HAL_GetTick();
    bmi270_sensor_data_t raw_data;

    // Record for target duration (180,000 ms = 3 min)
    while ((HAL_GetTick() - start_time) < duration_ms) {
        if (bmi270_sensor.readSensorData(&raw_data) == 0) {
            // Outputs raw 16-bit integer LSB values directly: ax,ay,az,gx,gy,gz,activity
            UART_printf("%d,%d,%d,%d,%d,%d,%d\r\n", 
                        (int)raw_data.acc_x, 
                        (int)raw_data.acc_y, 
                        (int)raw_data.acc_z, 
                        (int)raw_data.gyr_x, 
                        (int)raw_data.gyr_y, 
                        (int)raw_data.gyr_z, 
                        activity_id);
        }

        HAL_Delay(10); // 100 Hz sampling interval
    }

    UART_printf("--- COMPLETED %s ---\r\n", activity_name);
}

// Sequence through all 4 activity classes
void collect_all() {
    uint32_t duration_3_min = 3 * 60 * 1000; // 180,000 ms

    // Header matching your target format
    UART_printf("ax,ay,az,gx,gy,gz,activity\r\n");

    // 1 = Standing, 2 = Walking, 3 = Fast Walking, 4 = Running
    collect_activity(1, "Standing (0 mph)", duration_3_min);
    collect_activity(2, "Walking (2 mph)", duration_3_min);
    collect_activity(3, "Fast Walking (4 mph)", duration_3_min);
    collect_activity(4, "Running (7 mph)", duration_3_min);

    UART_printf("\r\n========================================\r\n");
    UART_printf("ALL DATA COLLECTION COMPLETE!\r\n");
    UART_printf("========================================\r\n");
}

int main(void) {
    // Resets all peripherals, Initializes the Flash interface and the Systick.
    HAL_Init();
    // Called once to initialize everything
    setup();

    // Print CSV Header on startup for new data logging session
    UART_printf("Timestamp_ms,Accel_X,Accel_Y,Accel_Z,Gyro_X,Gyro_Y,Gyro_Z\r\n");

        // MAIN DEAD LOOP: Uncomment the desired function to run either inference or data collection
        while (1) {
        //inference_loop();   //UNCOMMENT WHEN READY TO PERFORM INFERENCE
        
        collect_all();            //UNCOMMENT WHEN READY TO COLLECT RAW IMU DATA
            while(1) {
                HAL_Delay(1000);// Stay here after data collection is complete
            }
            
        }
}
