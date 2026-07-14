/* #include "board_setup.h"
#include "BMI270.h"

// TensorFlow Lite Micro Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"       
#include "tensorflow/lite/schema/schema_generated.h" 

#include "enhanced_activity_model.h"

// defining activity labels 
const char* activity_labels_cpp[] = {
    "Walking Upstairs",
    "Walking Downstairs",
    "Walking",
    "Sitting",
    "Standing",
    "Jogging"
};

// Pointers and objects used by TensorFlow Lite Micro to manage the model, 
// perform inference, resolve operations, and interact with input/output data.
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<6> op_resolver; 
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
// Memory allocation buffer arena for network tensor processing layers
// This is a common practice in memory-constrained embedded systems.
const int kTensorArenaSize = 25 * 1024; // 25 KB
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize]; // Aligned to 16 bytes for hardware vector speedup
// Scaling parameters to normalize the raw IMU data before feeding it into the neural network model.
const float g_mean[] = {1906.18630431, 5231.88101198, -4316.21812981, 10.75862685, -274.85723225, -202.32853567};
const float g_std[] = {6616.99734886, 12900.50103468, 5838.89261839, 4586.6168787, 6332.61625367, 6228.89244874};


//Error handler function for I2C initialization failures
void Error_Handler() {
    UART_printf("I2C Initialization Error! Check connections and configurations.\r\n");
    while (1) {
        // Stay here to indicate error
    }
}


// Declare I2C Handle globally
I2C_HandleTypeDef hi2c1; 
// Instantiate BMI270 class globally, passing the address of the I2C handle
BMI270 bmi270_sensor(&hi2c1);
// Function to initialize I2C for BMI270
void init_I2C_BMI270() {
    hi2c1.Instance = I2C1; // Use I2C1 peripheral
    hi2c1.Init.Timing = 0x00702991; // Example timing for 100kHz, adjust for your clock and desired speed
    hi2c1.Init.OwnAddress1 = 0; // Master mode, so no own address needed
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.OwnAddress2 = 0;
    hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    if (HAL_I2C_Init(&hi2c1) != HAL_OK) {
        // Error handling if I2C initialization fails
        Error_Handler(); // You should define an Error_Handler function
    }
    // Enable the Analog Noise Filter
    if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK) {
        Error_Handler();
    }
    // Enable the Digital Noise Filter (optional, depends on your needs)
    if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK) {
        Error_Handler();
    }
    UART_printf("I2C1 Initialized.\r\n");
}


// Placeholder for IMU data acquisition. Will now read from BMI270.
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
        UART_printf("Error reading BMI270 data! Using dummy data.\r\n");
        // Fallback to dummy data or error handling
        input_data[0] = 0.0f;
        input_data[1] = 0.0f;
        input_data[2] = 0.0f;
        input_data[3] = 0.0f;
        input_data[4] = 0.0f;
        input_data[5] = 0.0f;
    }
}

void setup() {
    // target specific initialization for TensorFlow Lite Micro
    tflite::InitializeTarget();
    // Initialize required pins
    init_GPIO_pins();
    init_UART2();
    init_TIM2();
    init_I2C_BMI270(); 
    UART_printf("Initializing TFLite Micro...\r\n");
    // Checks if the model's schema version matches the TFLite Micro library's expected version.
    model = tflite::GetModel(enhanced_activity_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch! Expected %d, got %d.",
                    TFLITE_SCHEMA_VERSION, model->version());
        return;
    }
    // Registers only "Fully Connected" and "Softmax" layers that my model uses
    op_resolver.AddFullyConnected(); 
    op_resolver.AddSoftmax();
    // Creates an interpreter instance, linking the model, op resolver, and the tensor arena.
    static tflite::MicroInterpreter static_interpreter(
        model, op_resolver, g_tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;

    // Allocate memory for all the model's tensors from the g_tensor_arena buffer
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed!");
        return;
    }

    // Gets pointers to the model's input and output tensors, which will be used to feed data in and read predictions out.
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    UART_printf("TFLite Micro initialized successfully!\r\n");

    // Initialize the BMI270 sensor
    if (bmi270_sensor.init() != 0) {
        UART_printf("Failed to initialize BMI270 sensor!\r\n");
    }
}



void loop() {
    // STM32 hardware SysTick Delay for 2.5s
    HAL_Delay(2500); 

    UART_printf("\r\nReading IMU data...\r\n");
    float raw_imu_data[6];
    read_imu_data(raw_imu_data);

    // Normalize incoming input features through the standard scaling parameters
    for (int i = 0; i < 6; ++i) {
        input_tensor->data.f[i] = (raw_imu_data[i] - g_mean[i]) / g_std[i];
    }

    // model performs its inference on the provided input data
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed!");
        return;
    }

    // retrieves the prediction probabilities from the model's output tensor
    float* output_scores = output_tensor->data.f;

    // inds the index of the output_scores array that has the highest value
    float max_score = -1.0f;
    int predicted_activity_index = -1;
    for (int i = 0; i < 6; ++i) {
        if (output_scores[i] > max_score) {
            max_score = output_scores[i];
            predicted_activity_index = i;
        }
    }

    // Broadcast results out to physical serial interface using native layout
    if (predicted_activity_index != -1) {
        UART_printf("Predicted Activity: %s (Score: %d.%04d)\r\n", 
                    activity_labels_cpp[predicted_activity_index],
                    (int)max_score, 
                    (int)((max_score - (int)max_score) * 10000));
    } else {
        UART_printf("Could not determine activity.\r\n");
    }
}



int main(void) {

    // Resets all peripherals, Initializes the Flash interface and the Systick.
    HAL_Init();
    
    // Called once to initialize everything
    setup();
    while (1) {
        // infinite loop to continuously read IMU data, perform inference, and output results
        loop();
    }
}
*/

#include "board_setup.h"
#include "bmi270.h"

// Declare I2C Handle globally (defined in board_setup.h or similar)
// For this example, we'll redefine it for clarity, assuming it's externed elsewhere.
I2C_HandleTypeDef hi2c1;

// Instantiate BMI270 class globally, passing the address of the I2C handle
BMI270 bmi270_sensor(&hi2c1);

// Error handler function for I2C initialization failures
void Error_Handler() {
    UART_printf("I2C Initialization Error! Check connections and configurations.\r\n");
    while (1) {
        // Stay here to indicate error
    }
}

// Function to initialize the I2C1 peripheral structure
void init_I2C_BMI270() {
    hi2c1.Instance = I2C1; // Use I2C1 peripheral
    
    // Timing configuration for 100kHz Standard Mode (Assuming a 16MHz or 80MHz internal clock)
    // Adjust this value based on your exact STM32L4 system clock if needed.
    hi2c1.Init.Timing = 0x00702991; 
    hi2c1.Init.OwnAddress1 = 0; 
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.OwnAddress2 = 0;
    hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    
    // This call automatically triggers HAL_I2C_MspInit underneath!
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


// --- MANDATORY HAL CALLBACK FUNCTION ---
// The HAL library automatically runs this function during HAL_I2C_Init() 
// to configure low-level clocks and physical pin hardware.
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

void setup() {
    // Initialize required pins
    init_GPIO_pins();
    init_UART2();
    
    HAL_Delay(100); // Short delay to ensure UART is ready
    UART_printf("Initializing I2C for BMI270...");
    init_I2C_BMI270();

    UART_printf("Initializing BMI270 sensor...");
    // Initialize the BMI270 sensor
    if (bmi270_sensor.init() != 0) {
        UART_printf("Failed to initialize BMI270 sensor! Check connections and sensor power.\r\n");
        while(1){}
    } else {
        UART_printf("BMI270 sensor initialized successfully!\r\n");
    }
}

void loop() {
    bmi270_sensor_data_t sensor_data;
    
    // Read sensor data
    if (bmi270_sensor.readSensorData(&sensor_data) == 0) {
        UART_printf("Acc: X=%.2f, Y=%.2f, Z=%.2f m/s^2 | Gyro: X=%.2f, Y=%.2f, Z=%.2f deg/s\r\n",
                    sensor_data.acc_x, sensor_data.acc_y, sensor_data.acc_z,
                    sensor_data.gyr_x, sensor_data.gyr_y, sensor_data.gyr_z);
    } else {
        UART_printf("Error reading BMI270 data!\r\n");
    }
    
    // Delay for a bit to make output readable
    HAL_Delay(100); // 100 ms delay
}

int main(void) {
    // Resets all peripherals, Initializes the Flash interface and the Systick.
    HAL_Init();

    // Called once to initialize everything
    setup();
    
    while (1) {
        // infinite loop to continuously read IMU data and print results
        loop();
    }
}