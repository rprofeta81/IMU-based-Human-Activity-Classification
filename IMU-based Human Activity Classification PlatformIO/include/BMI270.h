// BMI270.h
# ifndef BMI270_H
# define BMI270_H

# include <stdint.h>
// Include your microcontroller's HAL library for I2C communication
# include "stm32l4xx_hal.h"
# include "stm32l4xx_hal_i2c.h"

// Define BMI270 I2C address 
# define BMI270_I2C_ADDR_PRIMARY   (0x68 << 1) // 7-bit address 0x68, shifted for 8-bit R/W
# define BMI270_I2C_ADDR_SECONDARY (0x69 << 1) // 7-bit address 0x69, shifted for 8-bit R/W

// BMI270 Register addresses 
# define BMI270_CHIP_ID_ADDR       0x00
# define BMI270_ACC_CONF_ADDR      0x40 // ODR, Range, BW, etc.
# define BMI270_GYR_CONF_ADDR      0x42 // ODR, Range, BW, etc.
# define BMI270_PWR_CONF_ADDR      0x7C // Power modes
# define BMI270_CMD_ADDR           0x7E // Command register (soft reset, etc.)
# define BMI270_ACC_DATA_X_LSB_ADDR 0x0C // Start of accelerometer data registers
# define BMI270_GYR_DATA_X_LSB_ADDR 0x12 // Start of gyroscope data registers

typedef struct {
    float acc_x; // Accelerometer X-axis in m/s^2 or g
    float acc_y; // Accelerometer Y-axis
    float acc_z; // Accelerometer Z-axis
    float gyr_x; // Gyroscope X-axis in deg/s
    float gyr_y; // Gyroscope Y-axis
    float gyr_z; // Gyroscope Z-axis
} bmi270_sensor_data_t;

class BMI270 {
public:
    // Constructor, takes an I2C_HandleTypeDef pointer for communication
    BMI270(I2C_HandleTypeDef* hi2c);

    // Initialize the BMI270 sensor
    // Returns 0 on success, non-zero on failure
    int8_t init();

    // Read accelerometer data
    // Values will be updated in the provided pointers
    int8_t readAccelerometer(float* x, float* y, float* z);

    // Read gyroscope data
    // Values will be updated in the provided pointers
    int8_t readGyroscope(float* x, float* y, float* z);

    // Read all sensor data at once
    int8_t readSensorData(bmi270_sensor_data_t* data);

private:
    I2C_HandleTypeDef* _hi2c; // Pointer to I2C Handle
    uint8_t _device_address; // Storing the I2C device address

    // Generic I2C read/write functions using HAL
    int8_t i2c_read(uint8_t reg_addr, uint8_t *data, uint16_t len);
    int8_t i2c_write(uint8_t reg_addr, uint8_t *data, uint16_t len);

    // Helper function for BMI270 specific commands
    int8_t bmi270_set_command(uint8_t command);

    // Sensor specific configurations (e.g., scaling factors)
    float _accel_scale_factor; // Factor to convert raw accel data to m/s^2 or g
    float _gyro_scale_factor;  // Factor to convert raw gyro data to deg/s
};

# endif // BMI270_H