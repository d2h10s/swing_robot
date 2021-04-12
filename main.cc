#include "mbed.h"
#include "protocol.h"

//BufferedSerial xl430(D8, D2);

int main()
{
    Dynamixel xl430(0x01, PA_9, PA_10, PA_12);
    xl430.Goal_Position(10);
    while (true) {
    }
}