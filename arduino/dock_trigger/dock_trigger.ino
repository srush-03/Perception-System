/*
  arduino/dock_trigger/dock_trigger.ino
  Arduino Nano — monitors limit switch on charging bay.
  Sends "DOCKED\n" over serial (9600 baud) when drone docks.
  Sends "UNDOCKED\n" when drone lifts off.

  Wiring:
    Limit switch: one terminal → D2, other terminal → GND
    LED D13 lights when docked
*/
const int SWITCH_PIN  = 2;
const int LED_PIN     = 13;
const int DEBOUNCE_MS = 50;

bool last_state  = HIGH;
bool docked_sent = false;

void setup() {
  Serial.begin(9600);
  pinMode(SWITCH_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.println("ASCEND_DOCK_MONITOR_READY");
}

void loop() {
  bool current = digitalRead(SWITCH_PIN);
  if (current != last_state) {
    delay(DEBOUNCE_MS);
    current = digitalRead(SWITCH_PIN);
    if (current != last_state) {
      last_state = current;
      if (current == LOW) {
        Serial.println("DOCKED");
        digitalWrite(LED_PIN, HIGH);
        docked_sent = true;
      } else {
        Serial.println("UNDOCKED");
        digitalWrite(LED_PIN, LOW);
        docked_sent = false;
      }
    }
  }
  static unsigned long last_hb = 0;
  if (millis() - last_hb > 5000) {
    Serial.print("HB:"); Serial.println(docked_sent ? "DOCKED" : "READY");
    last_hb = millis();
  }
  delay(10);
}
