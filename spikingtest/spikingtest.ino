#include <Arduino.h>

const float increment     = 0.1f;                     // charge added on input spike
const float leak_ratio    = 0.01f;                    // fraction of increment lost per step
const float leak_amt      = increment * leak_ratio;   // absolute leak per step
const float v_rest        = 0.0f;                     // resting potential
const float threshold     = 1.0f;                     // spike threshold

const int   NUM_NEURONS        = 3;   // number of neurons
const int   INPUT_THRESHOLD    = 50;  // 0–100 chance (%) of an input spike
const int   TIME_STEP_MS       = 100; // delay between steps (ms)

float        v[NUM_NEURONS];              // membrane potentials
unsigned long t = 0;                      // time step counter

void setup() {
  Serial.begin(115200);
  while(!Serial);

  randomSeed(analogRead(A0));

  for(int i = 0; i < NUM_NEURONS; ++i) {
    v[i] = v_rest;
  }

  Serial.println("Step,V0,V1,V2,Spike0,Spike1,Spike2");
}

void loop() {
  bool fired[NUM_NEURONS];

  for(int i = 0; i < NUM_NEURONS; ++i) {
    bool spike_in = (random(0, 100) < INPUT_THRESHOLD);

    if(spike_in) {
      v[i] += increment;
    } else {
      v[i] = max(v[i] - leak_amt, v_rest);
    }

    if(v[i] >= threshold) {
      fired[i] = true;
      v[i] = v_rest;
    } else {
      fired[i] = false;
    }
  }

  Serial.print(t);
  for(int i = 0; i < NUM_NEURONS; ++i) {
    Serial.print(',');
    Serial.print(v[i], 3);
  }
  for(int i = 0; i < NUM_NEURONS; ++i) {
    Serial.print(',');
    Serial.print(fired[i] ? 1 : 0);
  }
  Serial.println();

  t++;
  delay(TIME_STEP_MS);
}
