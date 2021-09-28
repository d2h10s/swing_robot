#include <DynamixelWorkbench.h>
#include "JY901.h"
DynamixelWorkbench wb;

//FOR CONSTANT VARIABLES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SERIAL_BAUDRATE   115200
#define SERIAL_TIMEOUT    1000
#define AHRS_BAUDRATE     9600
#define AHRS_TIMEOUT      1000

#define MOTOR_BAUDRATE    115200

#define SERIAL_DEVICE     "1"     // Serial1
#define MX106_ID          1
#define MX106_CW_POS      2160
#define MX106_CCW_POS     1024
#define MX106_CURRENT     200
#define MX106_P_gain      400
#define MX106_I_gain      300
#define MX106_D_gain      4000

#define INNER_LED         14

#define frequency         (1./(0.661*2))
#define period            (1./(frequency))
#define half_period       ((period) / 2)
#define milli_half_period ((half_period)*1000)
// 1335 성공
// 1230 실패
// 1350 실패
bool is_MX106_on                  = false;
bool isOnline                     = false;
bool dir                          = false;
char logs[128]                    = {0};
unsigned long times               = 0;

float max_angle                   = 0;
float ahrs_roll_angle             = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(SERIAL_TIMEOUT);
  while(!Serial);
  sprintf(logs, "frequency is %fHz period is %fms", frequency, period*1000);
  Serial.println(logs);
  while (!motor_init());
  ahrs_init();
}

void loop() {
  // put your main code here, to run repeatedly:
  dir = !dir;
  times = millis();
  wb.goalPosition(MX106_ID, dir? MX106_CW_POS:MX106_CCW_POS);
  while (millis()-times <= milli_half_period){
    while(Serial2.available()) JY901.CopeSerialData(Serial2.read());
    ahrs_roll_angle = (float)JY901.stcAngle.Angle[0]/32768*180;
    max_angle = max_angle>abs(ahrs_roll_angle)? max_angle:abs(ahrs_roll_angle);
  }
  Serial.print(ahrs_roll_angle); Serial.print('\t'); Serial.println(max_angle);
}

void ahrs_init(){
  Serial2.begin(AHRS_BAUDRATE);
  Serial2.setTimeout(AHRS_TIMEOUT);
  for(int i = 0; i < 3; i++){
    while(Serial2.available()) JY901.CopeSerialData(Serial2.read());
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
  if (!is_MX106_on) { Serial.print("@set mode Failed!"); delay(100); return 0; }

  is_MX106_on = wb.writeRegister(MX106_ID, "Position_P_Gain", MX106_P_gain, &log);
  if (!is_MX106_on) { Serial.print("@Set P gain Failed!"); delay(100); return 0; }
  
  is_MX106_on = wb.writeRegister(MX106_ID, "Position_I_Gain", MX106_I_gain, &log);
  if (!is_MX106_on) { Serial.print("@Set I gain Failed!"); delay(100); return 0; }
  
  is_MX106_on = wb.writeRegister(MX106_ID, "Position_D_Gain", MX106_D_gain, &log);
  if (!is_MX106_on) { Serial.print("@Set D gain Failed!"); delay(100); return 0; }
  
  return 1;
}
