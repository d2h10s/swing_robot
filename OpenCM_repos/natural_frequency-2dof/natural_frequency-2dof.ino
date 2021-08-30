#include <DynamixelWorkbench.h>
DynamixelWorkbench wb;

//FOR CONSTANT VARIABLES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SAMPLING_TIME     25     // milli second
#define SERIAL_BAUDRATE   115200
#define MOTOR_BAUDRATE    115200
#define AHRS_BAUDRATE     115200

#define SERIAL_DEVICE     "1"     // Serial1
#define MX106_ID          1
#define MX106_CW_POS      2200
#define MX106_CCW_POS     1024
#define MX106_CURRENT     200

#define MX64_ID           2
#define MX64_CW_POS       2972
#define MX64_CCW_POS      2435
#define MX64_CURRENT      200

#define INNER_LED         14

#define milli_half_period 1335/2

// 1335 성공
// 1230 실패
// 1350 실패
bool is_MX106_on                  = false;
bool is_MX64_on                  = false;
bool isOnline                     = false;
char logs[128]                    = {0};
unsigned long times                   = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(180*1000);
  while(!Serial);
  sprintf(logs, "half period is %fms", milli_half_period);
  Serial.println(logs);
  while (!motor_init());
}

void loop() {
  // put your main code here, to run repeatedly:
  
  times = millis();
  wb.goalPosition(MX106_ID, MX106_CW_POS);
  wb.goalPosition(MX64_ID, MX64_CCW_POS);
  while (millis()-times <= milli_half_period);
  
  times = millis();
  wb.goalPosition(MX106_ID, MX106_CCW_POS);
  wb.goalPosition(MX64_ID, MX64_CW_POS);
  while (millis()-times <= milli_half_period);
}

int motor_init(){
  const char* log;
  isOnline = wb.init(SERIAL_DEVICE, MOTOR_BAUDRATE, &log);
  if (!isOnline) { Serial.print("@Port Open Failed!"); delay(100); return 0; }
  
  is_MX106_on = wb.ping(MX106_ID, &log);
  if (!is_MX106_on) { Serial.print("@Ping test Failed!"); delay(100); return 0; }

  is_MX106_on = wb.currentBasedPositionMode(MX106_ID, MX106_CURRENT, &log);
  if (!is_MX106_on) { Serial.print("@Set mode Failed!"); delay(100); return 0; }

  is_MX64_on = wb.ping(MX64_ID, &log);
  if (!is_MX64_on) { Serial.print("@Ping test Failed!"); delay(100); return 0; }

  is_MX64_on = wb.currentBasedPositionMode(MX64_ID, MX64_CURRENT, &log);
  if (!is_MX64_on) { Serial.print("@Set mode Failed!"); delay(100); return 0; }
  return 1;
}
