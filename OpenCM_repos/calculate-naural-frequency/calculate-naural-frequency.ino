
#define SERIAL_BAUDRATE   115200
#define AHRS_BAUDRATE     115200
#define RX_BUF_SIZE       128
#define TX_BUF_SIZE       128

//FOR AHRS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SOL               0x2A
#define EOL_CR            0x0D
#define EOL_LF            0x0A
#define SEP_CM            0x2C
#define AHRS_BUF_SIZE     128
#define AHRS_DATA_SIZE    7
#define AHRS_TIMEOUT      1000

//BUFFER VARIALBES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
String ahrs_buf                   = "";
float ahrs_data[AHRS_DATA_SIZE]   = {0};
char rx_buf[RX_BUF_SIZE]          = {0};
char tx_buf[TX_BUF_SIZE]          = {0};
float angle;
double roll;
float pre_angle = 0;
int state = 0;
int start_time = 0;
float pre_sign = 0;

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(1000);
  ahrs_init();
  while(!Serial);
  Serial.println("START");
}

void loop() {
  angle = 0;
  for(int i = 0 ; i < 3; i++){
    while(!getEulerAngles());
    angle += roll;
    delay(1);
  }
  angle /= 3;
  if ((pre_angle - angle)*pre_sign<0){
    Serial.println(millis()-start_time);
    start_time = millis();
  }
  pre_sign = pre_angle - angle;
  pre_angle = angle;
}

void ahrs_init(){
  digitalWrite(6, 0);
  delay(500);
  digitalWrite(6, 1);
  Serial2.begin(AHRS_BAUDRATE);
  Serial2.setTimeout(AHRS_TIMEOUT);
}

int getEulerAngles() {
  Serial2.write(SOL);
  ahrs_buf = Serial2.readStringUntil(EOL_LF);
  Serial2.flush();
  char ahrs_data_idx = 0;
  
  if (ahrs_buf[0] == SOL){
    String seg = "";
    for(int i = 1; i < ahrs_buf.length(); i++){
      if (ahrs_buf[i] == SEP_CM){
        ahrs_data[ahrs_data_idx++] = seg.toFloat();
        seg = "";
      }
      else if (ahrs_buf[i] == EOL_CR){
        ahrs_data[ahrs_data_idx++] = seg.toFloat();
        break;
      }
      else{
        seg += ahrs_buf[i];
      }
    }
  }
  else {
    Serial.println("@Failed to read AHRS");
    //Serial.println(ahrs_buf);
    delay(10);
    return 0;
  }
  //char buf[128];
  //sprintf(buf, "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",ahrs_data[0],ahrs_data[1],ahrs_data[2],ahrs_data[3],ahrs_data[4],ahrs_data[5],ahrs_data[6]);
  //Serial.println(buf);
  roll = ahrs_data[0];
  return 1;
}
