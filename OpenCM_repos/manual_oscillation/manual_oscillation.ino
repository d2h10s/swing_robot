#include <DynamixelWorkbench.h>
DynamixelWorkbench wb;

//FOR CONSTANT VARIABLES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SERIAL_BAUDRATE   115200
#define MOTOR_BAUDRATE    115200
#define SERIAL_DEVICE     "1"     // Serial1
#define MX106_ID          1
#define MX64_ID           2
#define MX106_CW_POS      2200
#define MX106_CCW_POS     1024
#define MX106_CURRENT     200
#define INNER_LED         14


bool is_MX106_on                  = false;
bool isOnline                     = false;
char logs[128]                   = {0};
void setup() {
  // put your setup code here, to run once:
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(180*1000);
  while(!Serial);
  Serial.println(logs);
  while (!motor_init());
}

void loop() {
  // put your main code here, to run repeatedly:
  wb.goalPosition(MX106_ID, MX106_CW_POS);
  while(!Serial.available()); Serial.read();
  wb.goalPosition(MX106_ID, MX106_CCW_POS);
  while(!Serial.available()); Serial.read();
}

int motor_init(){
  const char* log;
  is_MX106_on = wb.init(SERIAL_DEVICE, MOTOR_BAUDRATE, &log);
  if (!is_MX106_on) { Serial.print("@Port Open Failed!"); delay(100); return 0; }
  
  is_MX106_on = wb.ping(MX106_ID, &log);
  if (!is_MX106_on) { Serial.print("@Ping test Failed!"); delay(100); return 0; }

  is_MX106_on = wb.currentBasedPositionMode(MX106_ID, MX106_CURRENT, &log);
  if (!is_MX106_on) { Serial.print("@set mode Failed!"); delay(100); return 0; }
  
  return 1;
}
