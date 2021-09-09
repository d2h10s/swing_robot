#include <Wire.h>
#include "JY901.h"
/*
http://item.taobao.com/item.htm?id=43511899945
Test on mega2560.
JY901   mega2560
TX <---> 0(Rx)
*/
void setup() 
{
  Serial.begin(9600);  
  Serial2.begin(9600);
}

void loop() 
{
  Serial.print("Gyro:");Serial.print((float)JY901.stcGyro.w[0]/32768.*2000.); // degree/sec
  
  Serial.print("Angle:");Serial.print((float)JY901.stcAngle.Angle[0]/32768*180); // degree
  
  Serial.println("");
  delay(50);

  while (Serial2.available()) 
  {
    JY901.CopeSerialData(Serial2.read()); //Call JY901 data cope function
  }

}
