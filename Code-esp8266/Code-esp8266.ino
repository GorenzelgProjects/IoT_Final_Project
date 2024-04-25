//LIBARIES
#include <LiquidCrystal_I2C.h>

// Construct an LCD object and pass it the 
// I2C address, width (in characters) and
// height (in characters). Depending on the
// Actual device, the IC2 address may change. FOUND on address 0x27
LiquidCrystal_I2C lcd(0x27, 16, 2);



void setup() {
  //Constuct LCD screen (16 = widht, 2 = height *in chars)
  lcd.begin(16,2);
  lcd.init();
  

  // // Move the cursor characters to the right and
  // // zero characters down (line 1).
  // lcd.setCursor(5, 0);

  // // Print HELLO to the screen, starting at 5,0.
  // lcd.print("HELLO");

  // // Move the cursor to the next line and print
  // // WORLD.
  // lcd.setCursor(5, 1);      
  // lcd.print("WORLD");
  
}

void loop() {
  stringToLCD("Hello World",5000);



}


void stringToLCD(char* theText, int time){ //theText is what text that should be written on the screen and time is for how long it should be there in ms
  
  lcd.backlight(); // Turn on the backlight.

  
  //print text
  lcd.print(theText);


  delay(time); //time to show text
  lcd.noBacklight(); // Turn off the backlight.
  lcd.clear(); //clears screen


}