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
  pinMode(INNER_LED, OUTPUT);
  ahrs_init();
  while (!motor_init());
}

void loop() {
  if(Serial.available()){
    command = Serial.read();
    if (command == RST){
      Serial.print("STX,ACK!");
    }
    else if (command == GO_CW) {
      wb.goalPosition(MX106_ID, MX106_CW_POS);
      //wb.goalPosition(MX64_ID, MX64_CCW_POS);
    }
    else if (command == GO_CCW){
      wb.goalPosition(MX106_ID, MX106_CCW_POS);
      //wb.goalPosition(MX64_ID, MX64_CW_POS);
    }
    else if (command == ACQ){
      //while(!status()) motor_init();
      
      sprintf(tx_buf,"STX,ACQ,%f,%f,%f,%f!",
              ahrs_roll_angle, ahrs_roll_gyro, pos_buf[MX106_ID], vel_buf[MX106_ID]);
      Serial.print(tx_buf);
      delay(1);
    }
    else{
      sprintf(tx_buf, "@could not recognize bytes: %#X!", command);
      Serial.print(tx_buf);
      delay(1);
    }
  }
}


int status(){
  const char* log;
  is_MX106_on = wb.readRegister(MX106_ID, "Present_Velocity", vel_buf+MX106_ID, &log);
  if (!is_MX106_on) { sprintf(tx_buf, "@Failed to read velocity!"); Serial.print(tx_buf); delay(100); return 0; }
  
  is_MX106_on = wb.readRegister(MX106_ID, "Present_Position", pos_buf+MX106_ID, &log);
  if (!is_MX106_on) { sprintf(tx_buf, "@failed to read position!"); Serial.print(tx_buf); delay(100); return 0; }
  
  while(Serial2.available()) JY901.CopeSerialData(Serial2.read());
  ahrs_roll_gyro = (float)JY901.stcGyro.w[0]/32768.*2000.;
  ahrs_roll_angle = (float)JY901.stcAngle.Angle[0]/32768*180;
  
  return 1;
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
int motor_init(){
  const char* log;
  is_MX106_on = wb.init(SERIAL_DEVICE, MOTOR_BAUDRATE, &log);
  if (!is_MX106_on) { Serial.print("@Port Open Failed!"); delay(100); return 0; }
  
  is_MX106_on = wb.ping(MX106_ID, &log);
  if (!is_MX106_on) { Serial.print("@Ping test Failed!"); delay(100); return 0; }

  is_MX106_on = wb.currentBasedPositionMode(MX106_ID, MX106_CURRENT, &log);
  if (!is_MX106_on) { Serial.print("@Set mode Failed!"); delay(100); return 0; }

  is_MX106_on = wb.writeRegister(MX106_ID, "Position_P_Gain", MX106_P_gain, &log);
  if (!is_MX106_on) { Serial.print("@Set P gain Failed!"); delay(100); return 0; }
  
  is_MX106_on = wb.writeRegister(MX106_ID, "Position_I_Gain", MX106_I_gain, &log);
  if (!is_MX106_on) { Serial.print("@Set I gain Failed!"); delay(100); return 0; }
  
  is_MX106_on = wb.writeRegister(MX106_ID, "Position_D_Gain", MX106_D_gain, &log);
  if (!is_MX106_on) { Serial.print("@Set D gain Failed!"); delay(100); return 0; }
  /*
  is_MX64_on = wb.ping(MX64_ID, &log);
  if (!is_MX64_on) { Serial.print("@Ping test Failed!"); delay(100); return 0; }

  is_MX64_on = wb.currentBasedPositionMode(MX64_ID, MX64_CURRENT, &log);
  if (!is_MX64_on) { Serial.print("@Set mode Failed!"); delay(100); return 0; }
  */
  return 1;
}
