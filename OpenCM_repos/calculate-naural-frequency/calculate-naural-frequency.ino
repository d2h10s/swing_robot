#include "JY901.h"

#define SERIAL_BAUDRATE   115200
#define SERIAL_TIMEOUT    1000

//FOR AHRS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define AHRS_BAUDRATE     9600
#define AHRS_TIMEOUT      1000

//BUFFER VARIALBES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define sample            70

double angle, roll, pre_angle = 0, pre_sign = 0;
int state = 0;
int start_time = 0;
char logs[128];

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(SERIAL_TIMEOUT);
  ahrs_init();
  while(!Serial);
  Serial.println("START");
}

void loop() {
  angle = 0;
  for(int i = 0 ; i < sample; i++){
    while(Serial2.available()) JY901.CopeSerialData(Serial2.read());
    roll = (float)JY901.stcAngle.Angle[0]/32768*180;
    angle += roll;
    delay(1);
  }
  angle /= sample;
  if ((pre_angle - angle)*pre_sign<0.0){
    Serial.println(millis()-start_time);
    start_time = millis();
  }
  pre_sign = pre_angle - angle;
  pre_angle = angle;
  //Serial.println(angle);
}

void ahrs_init(){
  Serial2.begin(AHRS_BAUDRATE);
  Serial2.setTimeout(AHRS_TIMEOUT);
  for(int i = 0; i < 3; i++){
    while(Serial2.available()) JY901.CopeSerialData(Serial2.read());
    roll = (float)JY901.stcAngle.Angle[0]/32768*180;
  }
}
