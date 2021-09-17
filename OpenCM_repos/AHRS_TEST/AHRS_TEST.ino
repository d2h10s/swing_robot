#include <DynamixelWorkbench.h>
#include "JY901.h"
DynamixelWorkbench wb;

//FOR CONSTANT VARIABLES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SAMPLING_TIME     25     // milli second
#define SERIAL_BAUDRATE   115200
#define SERIAL_TIMEOUT    1000
#define MOTOR_BAUDRATE    115200
#define AHRS_BAUDRATE     9600
#define AHRS_TIMEOUT      1000

#define SERIAL_DEVICE     "1"     // Serial1
#define MX106_ID          1
#define MX106_CW_POS      2160
#define MX106_CCW_POS     1024
#define MX106_CURRENT     200
#define MX106_P_gain       400
#define MX106_I_gain       300
#define MX106_D_gain       4000

#define MX64_ID           2
#define MX64_CW_POS       2972
#define MX64_CCW_POS      2460
#define MX64_CURRENT      200

#define INNER_LED         14

//FOR PC COMMUNICATION>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define STX               0x02  // start of text
#define NLF               0x0A  // end of text
#define ACK               0x06  // Acknowlegement
#define NAK               0x15  // Negative Acknowledgement

#define ACQ               0x04  // pc acquires ovservation data
#define RST               0x05  // pc commands reset environment
#define GO_CW             0x70  // pc commands MX106 goes to min position
#define GO_CCW            0x71  // pc commands MX106 goes to max position

#define RX_BUF_SIZE       128
#define TX_BUF_SIZE       128


//BUFFER VARIALBES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

int32_t temp_buf[3]               = {0};
int32_t pos_buf[3]                = {0};
int32_t vel_buf[3]                = {0};
char rx_buf[RX_BUF_SIZE]          = {0};
char tx_buf[TX_BUF_SIZE]          = {0};
float ahrs_roll_angle             = 0;
float ahrs_roll_gyro              = 0;
uint8_t command                   = 0;
bool is_MX106_on                  = false;
bool is_MX64_on                   = false;
bool isOnline                     = false;

//MAIN PROGRAM>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(1000);
  ahrs_init();
}

void loop() {
  while(Serial2.available()) JY901.CopeSerialData(Serial2.read());
  ahrs_roll_gyro = (float)JY901.stcGyro.w[0]/32768.*2000.;
  ahrs_roll_angle = (float)JY901.stcAngle.Angle[0]/32768*180;
  Serial.print(ahrs_roll_angle+2.3); Serial.print('\t'); Serial.println(ahrs_roll_gyro);
  delay(10);
}


void ahrs_init(){
  Serial2.begin(AHRS_BAUDRATE);
  Serial2.setTimeout(AHRS_TIMEOUT);
  for(int i = 0; i < 3; i++){
    while(Serial2.available()) JY901.CopeSerialData(Serial2.read());
    ahrs_roll_gyro = (float)JY901.stcGyro.w[0]/32768.*2000.;
    ahrs_roll_angle = (float)JY901.stcAngle.Angle[0]/32768*180;
  }
}
