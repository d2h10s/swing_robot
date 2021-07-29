#include <DynamixelWorkbench.h>
DynamixelWorkbench wb;

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
#define AHRS_BUF_SIZE     128
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


//MAIN PROGRAM>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Serial.setTimeout(180*1000);
  Serial2.begin(AHRS_BAUDRATE);
  Serial2.setTimeout(180*1000);
  pinMode(6, OUTPUT);
  pinMode(INNER_LED, OUTPUT);
  //while(!Serial);
  ahrs_init();
  while (!motor_init());
}

void loop() {
  if(Serial.available()){
    command = Serial.read();
    if (command == RST){
      Serial.print("STX,ACK!");
      motor_init();
    }
    else if (command == GO_CW) {
      wb.goalPosition(MX106_ID, MX106_CW_POS);
    }
    else if (command == GO_CCW){
      wb.goalPosition(MX106_ID, MX106_CCW_POS);
    }
    else if (command == ACQ){
      while(!status());
      sprintf(tx_buf,"STX,ACQ,%f,%f,%f,%d,%d,%d!",
              ahrs_data[0], ahrs_data[4], ahrs_data[6], pos_buf[MX106_ID], vel_buf[MX106_ID], 0/*temp_buf[MX106_ID]*/);
      Serial.print(tx_buf);
      delay(1);
    }
    else{
      sprintf(tx_buf, "@could not recognize bytes: %#X!", command);
      Serial.print(tx_buf);
      delay(1);
    }
  }
}


int status(){
  const char* log;
  /*
  is_MX106_on = wb.readRegister(MX106_ID, "Present_Temperature", temp_buf+MX106_ID, &log);
  if (!is_MX106_on) {
    sprintf(tx_buf, "@Failed to read temperature!");
    Serial.print(tx_buf);
    delay(100);
    return 0;
  }
  */
  is_MX106_on = wb.readRegister(MX106_ID, "Present_Velocity", vel_buf+MX106_ID, &log);
  if (!is_MX106_on) {
    sprintf(tx_buf, "@Failed to read velocity!");
    Serial.print(tx_buf);
    delay(100);
    return 0;
  }
  is_MX106_on = wb.readRegister(MX106_ID, "Present_Position", pos_buf+MX106_ID, &log);
  if (!is_MX106_on) {
    sprintf(tx_buf, "@failed to read position!");
    delay(100);
    Serial.print(tx_buf);
    return 0;
  }
  while(getEulerAngles() != 1);
  return 1;
}

int motor_init(){
  const char* log;
  is_MX106_on = wb.init(SERIAL_DEVICE, MOTOR_BAUDRATE, &log);
  if (!is_MX106_on) {
    Serial.print("@Port Open Failed!");
    delay(100);
    return 0;
  }
  
  is_MX106_on = wb.ping(MX106_ID, &log);
  if (!is_MX106_on) {
    Serial.print("@Ping test Failed!");
    delay(100);
    return 0;
  }

  is_MX106_on = wb.currentBasedPositionMode(MX106_ID, MX106_CURRENT, &log);
  if (!is_MX106_on) {
    Serial.print("@set mode Failed!");
    delay(100);
    return 0;
  }
  
  return 1;
}

void ahrs_init(){
  digitalWrite(6, 0);
  delay(500);
  digitalWrite(6, 1);
  delay(500);
  Serial2.flush();
  Serial2.println("<sor0>"); // polling mode
  while (!Serial2.available()) while(Serial2.available()) Serial2.read();
  Serial2.println("<sot1>"); // print temperature
  while (!Serial2.available()) while(Serial2.available()) Serial2.read();
  Serial2.println("<soa4>"); // print acceleration
  while (!Serial2.available()) while(Serial2.available()) Serial2.read();
  Serial2.flush();
}


/* ahrs_buf [0]      [1]      [2]       [3]
 *          roll    pitch     yaw   temperature
 */
int getEulerAngles() {
  char temp = '\0';
  Serial2.write(SOL);
  delay(10);
  while (Serial2.available()) {
    temp = Serial2.read();
    ahrs_buf[ahrs_buf_idx++] = temp;
    if (ahrs_buf[0] == SOL && temp == EOL_LF) {
      ahrs_buf[ahrs_buf_idx - 1] = '\0';
      ahrs_buf[ahrs_buf_idx - 2] = ',';
      char *seg = strtok(ahrs_buf+1, ",");

      for (int i = 0; i < AHRS_DATA_SIZE; i++){
        ahrs_data[i] = atof(seg);
        seg = strtok(NULL, ",");
      }
      ahrs_buf_idx = 0;
    }
    if (ahrs_buf_idx >= AHRS_BUF_SIZE - 1) {
      Serial.println("@Failed to read AHRS");
      delay(10);
      ahrs_buf_idx = 0;
      return 0;
    }
  }
  return 1;
}
