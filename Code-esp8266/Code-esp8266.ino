//LIBARIES
//#include <LiquidCrystal_I2C.h> //LCD
#include <Wire.h>
#include <hd44780.h>                       // main hd44780 header
#include <hd44780ioClass/hd44780_I2Cexp.h> // i2c expander i/o class header

#include <SPI.h> //RFID
#include <MFRC522.h> //RFID
#include <Servo.h> //ServoMotor




#define MAX_UIDS 10   //Maximum number of storable UIDs
byte authorizedUIDs[MAX_UIDS][4];  //Array to store multiple UIDs
int storedUIDs = 0;  //Counter for the number of UIDs stored

#define RST_PIN D0          //For RFID
#define SS_PIN D8           //For RFID
#define BUTTON_PIN D4       //Pin connected to the button

MFRC522 mfrc522(SS_PIN, RST_PIN);  // Create MFRC522 instance
Servo servo;

// Construct an LCD object and pass it the 
// I2C address, width (in characters) and
// height (in characters). Depending on the
// Actual device, the IC2 address may change. FOUND on address 0x27
// LiquidCrystal_I2C lcd(0x27, 16, 2);
hd44780_I2Cexp lcd;
unsigned long lcdTimer = 0;


void setup() {
  Serial.begin(9600);
  SPI.begin();
  mfrc522.PCD_Init();
  servo.attach(D3);
  servo.write(0);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  

  // Serial.println(F("Scan PICC to see UID, SAK, type, and data blocks..."));

  //Constuct LCD screen (16 = widht, 2 = height *in chars)
  lcd.begin(16,2);

  stringToLCD("Hello World", 5000);
}

void loop() {
  
  if (access()) {
    
    rotate(90);  //Access granted, open the door
  }

  if (digitalRead(BUTTON_PIN) == LOW) {  //Check if the button is pressed
    //stringToLCD("Add New User", 5000);
    delay(500);
  
    addNewChip();  //Add a new RFID chip if pressed
  }
}

//Here is the functions used
bool access() {
  if (!mfrc522.PICC_IsNewCardPresent()) {
      return false;
  }

  if (!mfrc522.PICC_ReadCardSerial()) {
      return false;
  }

  if (isUIDInAccessList(mfrc522.uid.uidByte)) {
      Serial.println("Access Granted");
      mfrc522.PICC_HaltA();
      return true;
  } else {
      Serial.println("Access Denied");
      mfrc522.PICC_HaltA();
      return false;
  }
}

void rotate(int angle) {
  
  servo.write(angle);
  delay(3000); 
  
  servo.write(0);
}

void addNewChip() {
  if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial() && storedUIDs < MAX_UIDS) {
      // Add new UID to the list
      for (int i = 0; i < 4; i++) {
          authorizedUIDs[storedUIDs][i] = mfrc522.uid.uidByte[i];
      }
      storedUIDs++;  //Increment the count of stored UIDs
      Serial.println("New chip added for access.");
  }
}

bool isUIDInAccessList(byte* uid) {
  for (int i = 0; i < storedUIDs; i++) {
      bool match = true;
      for (int j = 0; j < 4; j++) {
          if (uid[j] != authorizedUIDs[i][j]) {
              match = false;
              break;
          }
      }
      if (match) {
          return true;
      }
  }
  return false;
}

void stringToLCD(char* theText, int time){ //theText is what text that should be written on the screen and time is for how long it should be there in ms
  lcdTimer = millis();
  
  
  lcd.backlight(); // Turn on the backlight.
  
  //print text
  lcd.print(theText);

  if (lcdTimer + time < millis()){
    lcd.noBacklight(); // Turn off the backlight.
    lcd.clear(); //clears screen
  }




}
