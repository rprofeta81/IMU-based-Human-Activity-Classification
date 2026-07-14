#ifndef BMI270_H
#define BMI270_H

#include <stdint.h>
#include "stm32l4xx_hal.h" // Ensures HAL types like I2C_HandleTypeDef are recognized

#define BMI270_I2C_ADDR_PRIMARY      (0x68 << 1)
#define BMI270_CHIP_ID_ADDR          0x00
#define BMI270_CMD_ADDR              0x7E
#define BMI270_PWR_CONF_ADDR         0x7C
#define BMI270_ACC_CONF_ADDR         0x40
#define BMI270_GYR_CONF_ADDR         0x42
#define BMI270_ACC_DATA_X_LSB_ADDR   0x0C
#define BMI270_GYR_DATA_X_LSB_ADDR   0x12

typedef struct {
    float acc_x;
    float acc_y;
    float acc_z;
    float gyr_x;
    float gyr_y;
    float gyr_z;
} bmi270_sensor_data_t;

class BMI270 {
public:
    BMI270(I2C_HandleTypeDef* hi2c);
    int8_t init();
    int8_t readAccelerometer(float* x, float* y, float* z);
    int8_t readGyroscope(float* x, float* y, float* z);
    int8_t readSensorData(bmi270_sensor_data_t* data);

private:
    I2C_HandleTypeDef* _hi2c;
    uint8_t _device_address;
    float _accel_scale_factor;
    float _gyro_scale_factor;

    int8_t i2c_read(uint8_t reg_addr, uint8_t *data, uint16_t len);
    int8_t i2c_write(uint8_t reg_addr, uint8_t *data, uint16_t len);
    int8_t bmi270_set_command(uint8_t command);
};

#endif