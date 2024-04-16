/*
  Three Wire LCD Interface using 74HC595 with ESP8266 NodeMCU
  www.circuits4you.com
*/
#include <ESP8266WiFi.h>

//Define 74HC595 Connections with arduio
const int Clock=D1; //D1
const int Data=D2; //D2
const int Latch=D3; //D3

char* line1 = "Circuits4You";
char* line2 = "Hello There";




void LCDinit(void);
void LCDSendByte(char d,char RS);
void LCDPrint(char Line,char string[16]);
//=========================================================
//              Setup
//=========================================================
void setup() {
  // initialize the digital pin as an output.
  pinMode(Data, OUTPUT);
  pinMode(Clock, OUTPUT);
  pinMode(Latch, OUTPUT);
  LCDinit(); //initalize LCD in 4-bit mode
}
//========================================================
//                Loop
//=========================================================
void loop() {

    LCDPrint(0,line1);
    LCDPrint(1,line2);
    delay(1000);
}
//=============================================================
//      Send Data or Command to LCD
//=============================================================
void LCDSendByte(char d,char RS)
{
  char dH,dL,temp; //Keep Data on upper nybble
  dH = d & 0xF0; //Get MSB
  dL = d & 0x0F;
  dL = d << 4; //Get LSB
  
  //Send MSB with E=clock
  temp=0;
  temp=dH | RS | 0x02; //MSB With RS+E bit
  ShiftData(temp); //Send MSB with E=0
  temp=0;
  temp=dH | RS; //MSB With RS bit
  ShiftData(temp);  //Send LSB with E=clock
  temp=0;
  temp=dL | RS | 0x02; //MSB With RS+E bit
  ShiftData(temp); //Send LSB with E=0
  temp=0;
  temp=dL | RS; //MSB With RS bit
  ShiftData(temp);
}
//=============================================
void ShiftData(char temp)
{
  int i;
  for(i=0;i<8;i++)
  {
    if((temp & 0x80)==0x80)   //Send 1-bit to shift register
    {digitalWrite(Data,HIGH);}
    else
    {digitalWrite(Data,LOW);}
    
    digitalWrite(Clock,LOW);
    digitalWrite(Clock,HIGH);
    temp=temp<<1;
  }
  //Latch the data
  digitalWrite(Latch,LOW);
  delay(1);
  digitalWrite(Latch,HIGH);
}
//=============================================================
//        LCD Display Initialization Function
//=============================================================
void LCDinit()
{
  int count;
  char t[]={0x43,0x03,0x03,0x02,0x28,0x01,0x0C,0x06,0x02,0x02};
  for (count = 0; count <= 9; count++)
  {
      LCDSendByte(t[count],0);  //Command RS=0
  }
}
//=============================================================
//Display Line on LCD at desired location Function
//=============================================================
void LCDPrint(char Line,char string[16])
{
  int len,count;
  if(Line==0) {
    LCDSendByte(0x80,0);  //Command RS=0 Move Cursor to Home
  }
  else {
    LCDSendByte(0xC0,0);  //Command RS=0 Move Cursor to Second Line
  }
  len = strlen(string);
  
  for (count=0;count<len;count++)
  {
    LCDSendByte(string[count],1); //Data RS=1
  }
}