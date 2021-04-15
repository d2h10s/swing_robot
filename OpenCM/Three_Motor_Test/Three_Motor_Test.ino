#include <DynamixelWorkbench.h>
DynamixelWorkbench dxl_wb;

#define BAUDRATE          9600
#define DEVICENAME        "1" // Serial1
#define DX1_id 3
void setup() {
  Serial.begin(9600);
  while(!Serial);
  
  const char* log;
  bool result;
  
  result = dxl_wb.init(DEVICENAME, BAUDRATE, &log);
  if (result) Serial.println("Port Open Success\n");
  else Serial.println(log), Serial.println("Port Open Failed\n");
  
  result = dxl_wb.ping(DX1_id, &log); // 필수, id
  if (result) Serial.println("Ping test Success\n");
  else Serial.println(log), Serial.println("Ping test Failed\n");
  
  dxl_wb.jointMode(DX1_id, 0, 0, &log);
  if (result) Serial.println("set mode Success\n");
  else Serial.println(log), Serial.println("set mode Failed\n");
}

void loop() {

  Serial.println("Go to 0");
  dxl_wb.goalPosition(DX1_id, 0);
  delay(2000);

  Serial.println("Go to 2000");
  dxl_wb.goalPosition(DX1_id, 2000);
  delay(2000);
}
