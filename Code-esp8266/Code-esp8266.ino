//LIBARIES
#include <Wire.h>
#include <hd44780.h>                       // main hd44780 header
#include <hd44780ioClass/hd44780_I2Cexp.h> // i2c expander i/o class header

#include <SPI.h> //RFID
#include <MFRC522.h> //RFID
#include <Servo.h> //ServoMotor

#include <WiFi.h>
#include "ThingSpeak.h"
#include "secrets.h"

//-----------------------------------------------------//
//              WIFI / ThingSpeak / Sound
//-----------------------------------------------------//
int led = 10;
int sound_digital = 19; //////MAYBE need to be changed to 9 which is pin SDD2
int sound_analog = A0;   

const int ARRAY_LENGTH = 64;
const int SAMPLE_TIME = 10;
const int RECORD = 5000;

// Time interval between each recordings (10 secs)
int startRecord = 5000;

// Time interval for the actual recording (5 secs)
int stopRecord = startRecord + 5000;

int recordStarted = 0;

unsigned long millisCurrent = 0;
unsigned long millisLast = 0;
unsigned long millisElapsed = 0;
unsigned long millisStart = 0;

int sampleBufferValue = 0;
int count_zero = 0;
int array_count = 0;

int RFID = 1; //VED IKKE OM VI SKAL BRUGE LÆNGERE

String resultData = "";

int currentSample = 1;
int inverseSample = -1;

unsigned long myChannelNumber = SECRET_CH_ID;
const char * myWriteAPIKey = SECRET_WRITE_APIKEY;

char ssid[] = SECRET_SSID;   // your network SSID (name)
char pass[] = SECRET_PASS;   // your network password


//-----------------------------------------------------//
//                 RFID / Servo
//-----------------------------------------------------//
#define MAX_UIDS 10   //Maximum number of storable UIDs
byte authorizedUIDs[MAX_UIDS][4];  //Array to store multiple UIDs
int storedUIDs = 0;  //Counter for the number of UIDs stored

#define RST_PIN D0          //For RFID
#define SS_PIN D8           //For RFID
#define BUTTON_PIN D4       //Pin connected to the button

MFRC522 mfrc522(SS_PIN, RST_PIN);  // Create MFRC522 instance
Servo servo;

//-----------------------------------------------------//
//                 LCD
//-----------------------------------------------------//
// Construct an LCD object and pass it the 
// I2C address, width (in characters) and
// height (in characters). Depending on the
// Actual device, the IC2 address may change. FOUND on address 0x27
// LiquidCrystal_I2C lcd(0x27, 16, 2);
hd44780_I2Cexp lcd;
unsigned long lcdTimer = 0;


void setup() {
  //-----------------------------------------------------//
  //                 WIFI / ThingSpeak / Sound
  //-----------------------------------------------------//
  WiFi.mode(WIFI_STA);
  ThingSpeak.begin(client);
  pinMode(led, OUTPUT);
  pinMode(sound_digital, INPUT);

  //-----------------------------------------------------//
  //                 RFID / Servo
  //-----------------------------------------------------//
  //Serial.begin(9600);
  SPI.begin();
  mfrc522.PCD_Init();
  servo.attach(D3);
  servo.write(0);
  pinMode(BUTTON_PIN, INPUT_PULLUP);


  //-----------------------------------------------------//
  //                 RFID / Servo
  //-----------------------------------------------------//
  Serial.begin(115200);
  pinMode(led, OUTPUT);
  pinMode(sound_digital, INPUT);



  ThingSpeak.begin(client);

  delay(100);
  

  
  //-----------------------------------------------------//
  //                 LCD
  //-----------------------------------------------------//
  lcd.begin(16,2);
  //stringToLCD("Hello World", 5000); test print text
}

void loop() {
  // Connect or reconnect to WiFi
  if (WiFi.status() != WL_CONNECTED) {
    Serial.print("Attempting to connect to SSID: ");
    Serial.println(SECRET_SSID);
    while (WiFi.status() != WL_CONNECTED) {
      WiFi.begin(ssid, pass); // Connect to WPA/WPA2 network. Change this line if using open or WEP network
      Serial.print(".");
      delay(5000);
    }
    Serial.println("\nConnected.");
  }else {
    int val_digital = digitalRead(sound_digital);
    int val_analog = analogRead(sound_analog);
    int result = val_digital;

    //checks RFID 
    // if (access()) {
    //   rotate(90);  //Access granted, open the door
    // }

    //add new user/card
    if (digitalRead(BUTTON_PIN) == LOW) {  //Check if the button is pressed
      delay(500);
    
      addNewChip();  //Add a new RFID chip if pressed
      stringToLCD("New User Added", 5000);
    }


    //Clean lcd screen
    if (lcdTimer < millis()){
      lcd.noBacklight(); // Turn off the backlight.
      lcd.clear(); //clears screen
    }

    if (val_digital == 1) {
      sampleBufferValue++;
    }
    //checks rfid then sound
    if (access()){
      stringToLCD("Card Accepted", 5000);//CARD ACCEPTED - u might want to play with time
      millisCurrent = millis();
      millisElapsed = millisCurrent - millisLast;
      if (recordStarted == 0) { 
        stopRecord = millisCurrent + RECORD;
        recordStarted = 1;
      }
      if (millisCurrent - millisStart > startRecord){
        digitalWrite (led, HIGH);
        if (millisElapsed > SAMPLE_TIME){

          if (sampleBufferValue*(-1) < 0){
            currentSample = (sampleBufferValue*(-1)+1)-111;
            inverseSample = currentSample * (-1);

          }
          else {
            currentSample = (sampleBufferValue);
            inverseSample = currentSample * (-1);
          }

          //---------------------------------------------------------------------------
          // DETTE SKAL KUN BRUGES TIL POTENTIOMETER  (MELLEM 1950 - 2000 på Y-aksen i serial-plotter)
          //Serial.println(val_analog);
          //---------------------------------------------------------------------------

          //---------------------------------------------------------------------------
          // HVIS DU VIL SE SELVE DATAEN REEL TID
          //Serial.println(currentSample);
          //Serial.println(inverseSample);
          //---------------------------------------------------------------------------

          sampleBufferValue = 0;
          millisLast = millisCurrent;

          if (inverseSample == 0){
            count_zero ++;
          }
          else {

            if (array_count < ARRAY_LENGTH){
              count_zero = count_zero * (-1);
              if (count_zero != 0){
                resultData.concat(String(count_zero));
                resultData.concat(String(","));
              };
              resultData.concat(String(inverseSample));
              resultData.concat(String(","));
            }

            count_zero = 0;
            array_count ++;
            array_count ++;
          }

        }
        if (millisCurrent - millisStart > stopRecord){
          millisStart = millisCurrent;

          //---------------------------------------------------------------------------
          // DETTE SKAL BRUGES TIL SELVE DATAINDSAMLING
          Serial.println(resultData);
          //---------------------------------------------------------------------------

          int httpCode = ThingSpeak.writeField(myChannelNumber, 1, resultData, myWriteAPIKey);

          if (httpCode == 200) {
            Serial.println("Channel write successful.");
          }
          else {
            Serial.println("Problem writing to channel. HTTP error code " + String(httpCode));
          }

          digitalWrite (led, LOW);
          count_zero = 0;
          array_count = 0;
          resultData = "";
          RFID = 0; //VED IKKE OM VI SKAL BRUGE LÆNGERE
          recordStarted = 0;
        }
      }
    }
  }
  
  

}//LOOP END






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

void openLock(int angle) {
  
  servo.write(angle);
  delay(3000); //MIGHT WANT TO INCREASE TIME
  
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
  lcdTimer = millis()+time;
  
  
  lcd.backlight(); // Turn on the backlight.
  
  //print text
  lcd.print(theText);
  lcd.setCursor(0,0);

  




}
