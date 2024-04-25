#include <SPI.h>
#include <MFRC522.h>
#include <Servo.h>

#define MAX_UIDS 10   //Maximum number of storable UIDs
byte authorizedUIDs[MAX_UIDS][4];  //Array to store multiple UIDs
int storedUIDs = 0;  //Counter for the number of UIDs stored

#define RST_PIN D0          //For RFID
#define SS_PIN D8           //For RFID
#define BUTTON_PIN D4       //Pin connected to the button

MFRC522 mfrc522(SS_PIN, RST_PIN);  // Create MFRC522 instance
Servo servo;

void setup() {
    Serial.begin(9600);
    SPI.begin();
    mfrc522.PCD_Init();
    servo.attach(0);
    servo.write(0);
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    Serial.println(F("Scan PICC to see UID, SAK, type, and data blocks..."));
}

void loop() {
    if (access()) {
        rotate(90);  //Access granted, open the door
    }

    if (digitalRead(BUTTON_PIN) == LOW) {  //Check if the button is pressed
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
