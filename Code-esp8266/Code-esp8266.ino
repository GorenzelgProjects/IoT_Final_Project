#include <ESP8266WiFi.h>
#include <Wire.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>
#include "ThingSpeak.h"
#include "secrets.h"

#include <hd44780.h>                     
#include <hd44780ioClass/hd44780_I2Cexp.h> 

#define RST_PIN D0
#define SS_PIN D8
#define BUTTON_PIN D4
#define LED_PIN 10    
#define SOUND_DIGITAL 9  
//D1 D2 used for i2cLCD

MFRC522 mfrc522(SS_PIN, RST_PIN); 
Servo servo;

WiFiClient client;

hd44780_I2Cexp lcd;
unsigned long lcdTimer = 0;

unsigned long myChannelNumber = SECRET_CH_ID;
const char* myWriteAPIKey = SECRET_WRITE_APIKEY;
const char* myReadAPIKey = SECRET_READ_APIKEY;
char ssid[] = SECRET_SSID;   // your network SSID (name)
char pass[] = SECRET_PASS;   // your network password

int storedUIDs = 0;
#define MAX_UIDS 10
byte authorizedUIDs[MAX_UIDS][4];

const int ARRAY_LENGTH = 64;
const int SAMPLE_TIME = 10;
const int RECORD = 5000;

int startRecord = 5000;
int stopRecord = startRecord + 5000;
int recordStarted = 0;

unsigned long millisCurrent = 0;
unsigned long millisLast = 0;
unsigned long millisElapsed = 0;
unsigned long millisStart = 0;

int sampleBufferValue = 0;
int count_zero = 0;
int array_count = 0;

String resultData = "";

int currentSample = 1;
int inverseSample = -1;

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  ThingSpeak.begin(client);

  pinMode(LED_PIN, OUTPUT);
  pinMode(SOUND_DIGITAL, INPUT);

  SPI.begin();
  mfrc522.PCD_Init();
  servo.attach(D3);  // Confirm pin number
  servo.write(0);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Serial.println("Setup complete.");
  lcd.begin(16,2);

}

void loop() {
  handleWiFi();
  handleRFIDandSound();
  clearLCD();

}

void handleWiFi() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Connecting to SSID: ");
    Serial.println(ssid);
    WiFi.begin(ssid, pass);
    while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      delay(500);
    }
    Serial.println("\nConnected.");
  }
}

void handleRFIDandSound() {
  int val_digital = digitalRead(SOUND_DIGITAL);
  int button_analog = digitalRead(BUTTON_PIN);

  Serial.println(button_analog);


  
  if (digitalRead(BUTTON_PIN) == LOW) {
    stringToLCD("Add User", 5000);
    delay(500);
    addNewChip();
  }

  if (val_digital == 1) {
    sampleBufferValue++;
  }

  if (access()) {
    Serial.println("Card Accepted");
    stringToLCD("Card Accepted", 3000);
    delay(3000);
    audio();
    delay(10000);
    int clap = 1;
    //int clap = ThingSpeak.readIntField(myChannelNumber,2,myReadAPIKey);
    Serial.println(clap);
    if (clap == 1){
      Serial.println("Clap Detected");
      stringToLCD("Open Lid", 3000);
      rotate(90);
    }
    else {
      Serial.println("No clap detected");
    }

    // Handle additional logic here
  } else {
    pass;
    //Serial.println("Card Denied");
  }
}

bool access() {
  if (!mfrc522.PICC_IsNewCardPresent() || !mfrc522.PICC_ReadCardSerial()) {
    return false;
  }
  if (isUIDInAccessList(mfrc522.uid.uidByte)) {
    Serial.println("Access Granted");
    mfrc522.PICC_HaltA();
    return true;
  }
  Serial.println("Access Denied");
  mfrc522.PICC_HaltA();
  return false;
}

void addNewChip() {
  if (mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial() && storedUIDs < MAX_UIDS) {
    for (int i = 0; i < 4; i++) {
      authorizedUIDs[storedUIDs][i] = mfrc522.uid.uidByte[i];
    }
    storedUIDs++;
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

void rotate(int angle) {
    for(int pos = 20; pos < angle; pos ++) {  // goes from 0 degrees to 180 degrees
                                    // in steps of 1 degree
      servo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(30);}
    //servo.write(angle);
    delay(3000);
    for(int pos = 20; pos < angle; pos ++) {  // goes from 0 degrees to 180 degrees
                                    // in steps of 1 degree
      servo.write(-pos);              // tell servo to go to position in variable 'pos'
      delay(30);}
    Serial.println("Servo should rotat");
    stringToLCD("Closed Lid", 3000);
}

void audio() {
  millisStart = millis();
  millisCurrent = millisStart;
  millisLast = millisCurrent;

  int responds = 0;
  Serial.println("START RECORDING");
  while (responds == 0){
    digitalWrite (LED_PIN, HIGH);
    int val_digital = digitalRead(SOUND_DIGITAL);
    //int val_analog = analogRead(sound_analog);
    int result = val_digital;

    if (val_digital == 1) {
      sampleBufferValue++;
    }
    delay(1);
    //Serial.println("SOUND");
    if (millisElapsed > SAMPLE_TIME){
      //Serial.println("SOUND");
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
      //millisLast = millisCurrent;

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

      millisLast = millisCurrent;
    }
    
    

    if (millisCurrent - millisStart > RECORD){
      millisStart = millisCurrent;
      responds = 1;
    }
    millisCurrent = millis();
    millisElapsed = millisCurrent - millisLast;
    //Serial.println(millisElapsed);
  

  }
    //---------------------------------------------------------------------------
    // DETTE SKAL BRUGES TIL SELVE DATAINDSAMLING
    digitalWrite (LED_PIN, LOW);
    Serial.println("DONE RECORDING - SENDING TO THINKSPEAK");
    Serial.println(resultData);
    //---------------------------------------------------------------------------

    int httpCode = ThingSpeak.writeField(myChannelNumber, 1, resultData, myWriteAPIKey);

    if (httpCode == 200) {
      Serial.println("Channel write successful.");
    }
    else {
      Serial.println("Problem writing to channel. HTTP error code " + String(httpCode));
    }

    
    count_zero = 0;
    array_count = 0;
    resultData = "";
    //RFID = 0;
    recordStarted = 0;
}


void stringToLCD(char* theText, int time){ 
  lcdTimer = millis()+time;
  
  lcd.setCursor(0,0);
  lcd.backlight(); // Turn on the backlight.
  
  //print text
  lcd.print(theText);
  
}

void clearLCD(){
  if (lcdTimer < millis()){
      lcd.noBacklight(); // Turn off the backlight.
      lcd.clear(); //clears screen
  }
}
