#ifndef protocol_H
#define protocol_H
#include "mbed.h"

namespace INST{
    enum {Ping, Read, Write, Reg_Write, Action, Factory_Reset, Reboot=0x08, Clear=0x10, Status=055, Sync_Read=0x82, Sync_Write=0x83, Bulk_Read=0x92, Bulk_Write=0x93};
}

class Dynamixel{
private:
    uint8_t ID, LENGTH;
    /// Header 1	Header 2	Header 3	Reserved	Packet ID	Length_L	Length_H	Instruction	Param	Param	Param	CRC_L	CRC_H
    uint8_t send_buffer[40] = {0xFF, 0xFF, 0xFD, 0x00, 0, 0, 0, 0,};
    /// ADDR_L ADDR_H Params ...
    uint8_t params[40] = {0,};
    DigitalOut enb;
    BufferedSerial ser;

    void send_packet(uint8_t inst, uint8_t length);
    uint16_t update_crc(uint16_t crc_accum, uint8_t *data_blk_ptr, uint16_t data_blk_size);

public:
    /**
    *   @param id i2c communication id
    *   @param TX Transmit Pin
    *   @param RX Recieve Pin
    *   @param ENB TX-RX Switch Pin
    **/
    Dynamixel(uint8_t id, PinName TX, PinName RX, PinName ENB): ID(id), ser(TX, RX, 57600), enb(ENB) {send_buffer[4] = ID;}
    void Goal_Position(uint16_t position);
};

#endif

