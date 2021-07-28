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
#define AHRS_BUF_SIZE     64
#define AHRS_DATA_SIZE    7

//BUFFER VARIALBES>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
char ahrs_buf[AHRS_BUF_SIZE]      = {0};
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
  Serial2.begin(115200);
  ahrs_init();
  while(!Serial);
}

void loop() {
  // put your main code here, to run repeatedly:
  while(getEulerAngles() != 1);
  
  delay(1000);
}
void ahrs_init(){
  digitalWrite(6, 0);
  delay(500);
  digitalWrite(6, 1);
  delay(500);
  Serial2.println("<sor0>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<sof1>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<sog0>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<sot1>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
  Serial2.println("<soa4>");
  delay(1000);
  while (Serial2.available()) Serial2.read();
}

int getEulerAngles() {
  digitalWrite(INNER_LED, 0);
  Serial2.write(SOL);
  while (Serial2.available()) {
    ahrs_buf[ahrs_buf_idx++] = Serial2.read();
    Serial.print(ahrs_buf[ahrs_buf_idx-1]);
    if (ahrs_buf[0] == SOL && ahrs_buf[ahrs_buf_idx - 1] == EOL_LF) {
      Serial.println(ahrs_buf);
      ahrs_buf[ahrs_buf_idx - 1] = ',';
      char *seg = strtok(ahrs_buf + 1, ",");
      for (int i = 0; i < AHRS_DATA_SIZE; i++){
        Serial.print(seg); Serial.print(':');
        ahrs_data[i] = atof(seg);
        seg = strtok(NULL, ",");
        Serial.print(ahrs_data[i]); Serial.print('\t');
      }
      Serial.print('\n');
      ahrs_buf_idx = 0;
    }
    if (ahrs_buf_idx >= AHRS_BUF_SIZE - 1) {
      Serial.println("@Failed to read AHRS");
      return 0;
    }
  }
  digitalWrite(INNER_LED, 1);
  return 1;
}

int getEulerAngles2() {
  digitalWrite(INNER_LED, 0);
  Serial2.write(SOL);
  delay(1);
  while (Serial2.available()) {
    ahrs_buf[ahrs_buf_idx++] = Serial2.read();
    Serial.print(ahrs_buf[ahrs_buf_idx++]);
    if (ahrs_buf[ahrs_buf_idx - 1] == EOL_LF) ahrs_buf[ahrs_buf_idx - 1] = ',';

    if (ahrs_buf[ahrs_buf_idx - 1] == SOL){
      char *seg = strtok(ahrs_buf, ",");

      for (int i = 0; i < AHRS_DATA_SIZE; i++){
        ahrs_data[i] = atof(seg);
        seg = strtok(NULL, ",");
      }
      ahrs_buf_idx = 0;
    }
    if (ahrs_buf_idx >= AHRS_BUF_SIZE - 1) {
      Serial.println("@Failed to read AHRS");
      return 0;
    }
  }
  digitalWrite(INNER_LED, 1);
  return 1;
}
