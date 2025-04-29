#include "arducam_dvp.h"
#include <Wire.h>
#include <Arduino_PortentaBreakout.h>
#include "stm32h7xx_hal.h"
#include "weights.h"

static float increment   = 0.1f;
static float leak_ratio  = 0.01f;
static float leak_amt    = increment * leak_ratio;
static float threshold   = 1.0f;
static constexpr float v_rest = 0.0f;

#define IMG_W     320
#define IMG_H     240

#define N_BINS    64   // inputs
#define N_H       64   // hidden
#define N_O       16    // outputs
#define N_LBL     2
#define NPG       (N_O / N_LBL)

#define NUM_STEPS 15

#define ARDUCAM_CAMERA_OV767X
#ifdef ARDUCAM_CAMERA_OV767X
  #include "OV7670/ov767x.h"
  OV7670 ov767x; Camera cam(ov767x);
  #define IMAGE_MODE CAMERA_RGB565
#else
  #error "Select your camera model"
#endif

FrameBuffer fb;
TIM_HandleTypeDef htim1;

static const uint8_t FRAME_END_MARKER[4] = { 0xDE,0xAD,0xBE,0xEF };
static const uint8_t INFER_HDR[4]        = { 0xCA,0xFE,0xBA,0xBE };
const uint8_t REQUEST_BYTE = 0x01;

void setupTimer1ForPWM() {
  __HAL_RCC_TIM1_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  GPIO_InitTypeDef g = {};
  g.Pin       = GPIO_PIN_8;
  g.Mode      = GPIO_MODE_AF_PP;
  g.Pull      = GPIO_NOPULL;
  g.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
  g.Alternate = GPIO_AF1_TIM1;
  HAL_GPIO_Init(GPIOA, &g);

  htim1.Instance = TIM1;
  htim1.Init.Prescaler     = 1;
  htim1.Init.CounterMode   = TIM_COUNTERMODE_UP;
  htim1.Init.Period        = 6;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  HAL_TIM_PWM_Init(&htim1);

  TIM_OC_InitTypeDef s = {};
  s.OCMode     = TIM_OCMODE_PWM1;
  s.Pulse      = 4;  // 50% duty @ 30 MHz
  s.OCPolarity = TIM_OCPOLARITY_HIGH;
  s.OCFastMode = TIM_OCFAST_ENABLE;
  HAL_TIM_PWM_ConfigChannel(&htim1, &s, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
}

void blinkLED(uint32_t cnt=0xFFFFFFFF, uint32_t d=50) {
  while(cnt--) {
    digitalWrite(LED_BUILTIN, LOW);  delay(d);
    digitalWrite(LED_BUILTIN, HIGH); delay(d);
  }
}

const float* W1ptr = W1_data; 
const float* W2ptr = W2_data; 

int infer_and_groups(const float x[N_BINS], int grp[N_LBL]) {
  static bool seeded = false;
  if(!seeded){
    randomSeed(micros());
    seeded = true;
  }

  float v_h[N_H], v_o[N_O];
  int   os_h[N_H] = {0}, os_o[N_O] = {0};

  for(int i = 0; i < N_H; ++i) v_h[i] = v_rest;
  for(int i = 0; i < N_O; ++i) v_o[i] = v_rest;

  for(int t = 0; t < NUM_STEPS; ++t) {
    for(int p = 0; p < N_BINS; ++p) {
      bool spike_in = (random(0,10000) < x[p] * 10000.0f);

      if(!spike_in) {
        for(int h=0; h<N_H; ++h)
          v_h[h] = fmaxf(v_h[h] - leak_amt, v_rest);
      } else {
        int gs  = N_H / N_BINS;
        int rem = N_H - gs * N_BINS;
        int extra   = (p < rem) ? 1 : 0;
        int start_h = p*gs + min(p,rem);
        int end_h   = start_h + gs + extra;
        for(int h=start_h; h<end_h; ++h)
          v_h[h] += increment * W1ptr[h];
      }

      bool fired_h[N_H];
      bool any_h = false;
      for(int h=0; h<N_H; ++h) {
        if(v_h[h] >= threshold) {
          fired_h[h] = true;
          os_h[h]   += 1;
          v_h[h]     = v_rest;
          any_h      = true;
        } else {
          fired_h[h] = false;
        }
      }

      if(any_h) {
        for(int o=0; o<N_O; ++o) {
          float acc = 0.0f;
          for(int h=0; h<N_H; ++h)
            if(fired_h[h])
              acc += W2ptr[o * N_H + h];
          v_o[o] += increment * acc;
        }
      } else {
        for(int o=0; o<N_O; ++o)
          v_o[o] = fmaxf(v_o[o] - leak_amt, v_rest);
      }

      for(int o=0; o<N_O; ++o) {
        if(v_o[o] >= threshold) {
          os_o[o] += 1;
          v_o[o]   = v_rest;
        }
      }
    }
  }

  grp[0] = grp[1] = 0;
  for(int j = 0; j < N_LBL; ++j)
    for(int k = 0; k < NPG; ++k)
      grp[j] += os_o[j * NPG + k];

  return (grp[1] > grp[0]) ? 1 : 0;
}

void setup(){
  Serial.begin(12500000);
  __disable_irq();
    setupTimer1ForPWM();
  __enable_irq();

  pinMode(LED_BUILTIN, OUTPUT);
  if(!cam.begin(CAMERA_R320x240, IMAGE_MODE, 30)) blinkLED();
  blinkLED(5);
}

void loop(){
  if(Serial.available()){
    int c = Serial.peek();
    if(c == 'H'){
      String line = Serial.readStringUntil('\n');
      if(line.startsWith("HP ")){
        float inc  = line.substring(3, line.indexOf(' ',3)).toFloat();
        float leak = line.substring(line.indexOf(' ',3)+1).toFloat();
        increment  = inc;
        leak_ratio = leak;
        leak_amt   = increment * leak_ratio;
        Serial.println("OK");
      } else Serial.println("ERR");
      return;
    }
    if(c == 'B'){
      String line = Serial.readStringUntil('\n');
      float feats[N_BINS];
      char buf[256];
      line.toCharArray(buf, sizeof(buf));
      int idx = 0;
      char *tok = strtok(buf + 5, ",");
      while(tok && idx < N_BINS){
        feats[idx++] = atof(tok);
        tok = strtok(NULL, ",");
      }
      int grp[N_LBL];
      int lbl = infer_and_groups(feats, grp);

      Serial.print("GROUPS:");
      Serial.print(grp[0]);
      Serial.print(',');
      Serial.println(grp[1]);

      Serial.write(INFER_HDR, sizeof(INFER_HDR));
      Serial.write((uint8_t*)&lbl, 1);
      Serial.flush();
      return;
    }
  }

  if(Serial.read() != REQUEST_BYTE){
    blinkLED(1,10);
    return;
  }

  if(cam.grabFrame(fb, 3000) == 0){
    Serial.write(fb.getBuffer(), cam.frameSize());
    Serial.write(FRAME_END_MARKER, sizeof(FRAME_END_MARKER));
  } else {
    blinkLED(20,100);
    delay(1000);
  }
}
