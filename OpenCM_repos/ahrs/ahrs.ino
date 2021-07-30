//FOR CONSTANT VARIABLES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SERIAL_BAUDRATE   115200
#define MOTOR_BAUDRATE    115200
#define AHRS_BAUDRATE     115200
#define SERIAL_DEVICE     "1"     // Serial1
#define MX106_ID          1
#define MX64_ID           2
#define SAMPLING_TIME     25     // milli second
#define MX106_CW_POS      2200
#define MX106_CCW_POS     1024
#define MX106_CURRENT     200
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

//FOR AHRS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define SOL               0x2A
#define EOL_CR            0x0D
#define EOL_LF            0x0A
#define SEP_CM            0x2C
#define AHRS_BUF_SIZE     64
#define AHRS_DATA_SIZE    7

//BUFFER VARIALBES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
String  ahrs_buf;
float ahrs_data[AHRS_DATA_SIZE]   = {0};
int32_t temp_buf[3]               = {0};
int32_t pos_buf[3]                = {0};
int32_t vel_buf[3]                = {0};
char ahrs_temp                    = 0;
char ahrs_buf_idx                 = 0;
char rx_buf[RX_BUF_SIZE]          = {0};
char tx_buf[TX_BUF_SIZE]          = {0};

uint8_t command                   = 0;
bool is_MX106_on                  = false;
bool isOnline                     = false;


void setup() {
  // put your setup code here, to run once:
  pinMode(INNER_LED, OUTPUT);
  Serial.begin(115200);
  ahrs_init();
  while(!Serial);
  Serial.println("loop start");
}

void loop() {
  // put your main code here, to run repeatedly:
  while(getEulerAngles() != 1);
  
  delay(100);
}
void ahrs_init(){
  Serial.println("ahrs init start");
  digitalWrite(6, 0);
  delay(500);
  digitalWrite(6, 1);
  Serial2.begin(115200);
  Serial2.setTimeout(1000);
  Serial2.flush();
  Serial2.println("<sor0>"); // polling mode
  while (!Serial2.available()); while(Serial2.available()) Serial.write(Serial2.read());
  Serial2.println("<sot1>"); // print temperature
  while (!Serial2.available()); while(Serial2.available()) Serial.write(Serial2.read());
  Serial2.println("<soa4>"); // print acceleration
  while (!Serial2.available()); while(Serial2.available()) Serial.write(Serial2.read());
  Serial2.flush();
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
    Serial.println(ahrs_buf);
    delay(10);
    return 0;
  }
  char buf[128];
  sprintf(buf, "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",ahrs_data[0],ahrs_data[1],ahrs_data[2],ahrs_data[3],ahrs_data[4],ahrs_data[5],ahrs_data[6]);
  Serial.println(buf);
  return 1;
}
//int getEulerAngles() {
//  Serial.println("get angle");
//  Serial2.write(SOL);
//  String s = Serial2.readStringUntil(EOL_LF);
//  Serial.println(s);
//  /*
//  Serial.print("받기 전:"); Serial.println(Serial2.available());
//  Serial2.write(SOL);
//  while(temp != SOL){
//    temp = Serial2.read();
//    Serial.print((int)temp);
//    Serial.print(' ');
//  }
//  Serial.print("받은 후:"); Serial.println(Serial2.available());
//  
//  while(Serial2.available()) Serial.write(temp = Serial2.read());
//  Serial2.flush();
//  */
//  /*
//  char temp = '\0';
//  Serial2.write(SOL);
//  delay(10);
//  while (1) {
//    temp = Serial2.read();
//    //Serial.write(temp);
//    ahrs_buf[ahrs_buf_idx++] = temp;
//    if (ahrs_buf[0] == SOL && temp == EOL_LF) {
//      Serial.print(ahrs_buf);
//      ahrs_buf[ahrs_buf_idx - 1] = '\0';
//      ahrs_buf[ahrs_buf_idx - 2] = ',';
//      char *seg = strtok(ahrs_buf+1, ",");
//
//      for (int i = 0; i < AHRS_DATA_SIZE; i++){
//        ahrs_data[i] = atof(seg);
//        seg = strtok(NULL, ",");
//      }
//      Serial.println("end!!");
//      ahrs_buf_idx = 0;
//    }
//    if (ahrs_buf_idx >= AHRS_BUF_SIZE - 1) {
//      Serial.println("@Failed to read AHRS");
//      delay(10);
//      ahrs_buf_idx = 0;
//      return 0;
//    }
//  }
//  char buf[128];
//  //sprintf(buf, "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",ahrs_data[0],ahrs_data[1],ahrs_data[2],ahrs_data[3],ahrs_data[4],ahrs_data[5],ahrs_data[6]);
//  //Serial.println(buf);
//  */
//  return 1;
//}
