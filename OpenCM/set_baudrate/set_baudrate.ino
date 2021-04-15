#include <DynamixelWorkbench.h>
DynamixelWorkbench dxl_wb;

#define DXL_ID            1
#define PREVIOUS_BAUDRATE 115200
#define POST_BAUDRATE     9600
#define DEVICENAME        "1"

void setup() {
  const char* log;
  int result;
  
  Serial.begin(9600);
  while(!Serial);
  
  result = dxl_wb.init(DEVICENAME, PREVIOUS_BAUDRATE, &log);
  dxl_wb.ping(DXL_ID);
  if (result) Serial.println("Port open succeed");
  else {
    Serial.println(log);
    Serial.println("Port open failed");
  }
  
  
  result = dxl_wb.setBaudrate(POST_BAUDRATE, &log);
  if (result) Serial.println("set succeed");
  else {
    Serial.println(log);
    Serial.println("set failed");
  }
}

void loop() {
  // put your main code here, to run repeatedly:

}
