#define BAUDRATE 115200

void setup() {
  // put your setup code here, to run once:
  pinMode(6, OUTPUT);
  digitalWrite(6, 0);
  delay(10);
  digitalWrite(6, 1);
  Serial.begin(BAUDRATE);
  Serial2.begin(BAUDRATE);
  while(!Serial);
  delay(1000);
  Serial.flush();
  Serial2.flush();
  Serial2.println("<sor0>"); // polling mode
  while (!Serial2.available()) while(Serial2.available()) Serial2.read();
  Serial2.println("<sot1>"); // print temperature
  while (!Serial2.available()) while(Serial2.available()) Serial2.read();
  Serial2.println("<soa4>"); // print acceleration
  while (!Serial2.available()) while(Serial2.available()) Serial2.read();
  Serial.flush();
  Serial2.flush();
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()) Serial2.write(Serial.read());
  if(Serial2.available()) Serial.write(Serial2.read());
}
