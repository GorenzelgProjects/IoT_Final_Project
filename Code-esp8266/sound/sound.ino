#include <WiFi.h>
#include "ThingSpeak.h"
#include "secrets.h"

int led = 18;
int sound_digital = 19;
int sound_analog = 36;

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

int RFID = 1;

String resultData = "";

int currentSample = 1;
int inverseSample = -1;

unsigned long myChannelNumber = SECRET_CH_ID;
const char * myWriteAPIKey = SECRET_WRITE_APIKEY;

char ssid[] = SECRET_SSID;   // your network SSID (name)
char pass[] = SECRET_PASS;   // your network password

WiFiClient  client;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(led, OUTPUT);
  pinMode(sound_digital, INPUT);

  WiFi.mode(WIFI_STA);

  ThingSpeak.begin(client);

  delay(100);
}

void loop() {
  // put your main code here, to run repeatedly:

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
  }
  else {
    int val_digital = digitalRead(sound_digital);
    int val_analog = analogRead(sound_analog);
    int result = val_digital;

    if (val_digital == 1) {
      sampleBufferValue++;
    }

    if (RFID == 1){
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
          RFID = 0;
          recordStarted = 0;
        }
      }
    }
  }
}
