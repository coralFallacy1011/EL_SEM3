// Pin definitions for 74HC595
#define DATA_PIN  13   // Serial data input
#define LATCH_PIN 12   // Register clock
#define CLOCK_PIN 14   // Serial clock

// Matrix dimensions
#define ROWS 8
#define COLS 16
#define RGB_BITS 3     // Bits per color (R,G,B)
#define TOTAL_RGB_BITS (COLS * RGB_BITS)  // 48 bits for RGB control
#define TOTAL_SHIFT_REGISTERS 7  // Total number of 74HC595s

// Function prototypes
void setupPins();
void shiftOut48Bits(uint8_t* data);
void updateMatrix(uint8_t x, uint8_t y, uint8_t r, uint8_t g, uint8_t b);
void clearMatrix();

// Global variables for shift register data
uint8_t shiftRegData[7];  // 7 bytes for shift registers (48 bits RGB + 8 bits row)

void setup() {
  setupPins();
  clearMatrix();
  
  // Example coordinates to turn off specific LEDs (hardcoded for now)
  // Format: x, y, r, g, b (0 = LED on, 1 = LED off for common anode)
  updateMatrix(0, 0, 1, 1, 1);  // Turn off LED at (0,0)
  updateMatrix(1, 1, 1, 1, 1);  // Turn off LED at (1,1)
  updateMatrix(2, 3, 1, 1, 1);  // Turn off LED at (2,3)
  updateMatrix(5, 10, 0, 0, 0);
}

void loop() {
  // Main loop - currently just maintaining the display
  // You can add animations or other updates here
}

void setupPins() {
  pinMode(DATA_PIN, OUTPUT);
  pinMode(LATCH_PIN, OUTPUT);
  pinMode(CLOCK_PIN, OUTPUT);
  
  digitalWrite(DATA_PIN, LOW);
  digitalWrite(LATCH_PIN, LOW);
  digitalWrite(CLOCK_PIN, LOW);
}

void shiftOut48Bits(uint8_t* data) {
  digitalWrite(LATCH_PIN, LOW);
  
  // Shift out all bytes
  for(int i = TOTAL_SHIFT_REGISTERS - 1; i >= 0; i--) {
    shiftOut(DATA_PIN, CLOCK_PIN, MSBFIRST, data[i]);
  }
  
  digitalWrite(LATCH_PIN, HIGH);
  delayMicroseconds(1);
  digitalWrite(LATCH_PIN, LOW);
}

void updateMatrix(uint8_t x, uint8_t y, uint8_t r, uint8_t g, uint8_t b) {
  if(x >= COLS || y >= ROWS) return;  // Bounds check
  
  // Calculate bit positions
  uint8_t colOffset = x * RGB_BITS;  // Each column uses 3 bits (RGB)
  uint8_t byteIndex = colOffset / 8;
  uint8_t bitOffset = colOffset % 8;
  
  // Update RGB bits for the column
  // Note: For common anode, 1 = LED off, 0 = LED on
  
  // Red bit
  if(r) {
    shiftRegData[byteIndex] |= (1 << (7 - bitOffset));
  } else {
    shiftRegData[byteIndex] &= ~(1 << (7 - bitOffset));
  }
  
  // Green bit
  if(g) {
    shiftRegData[byteIndex] |= (1 << (6 - bitOffset));
  } else {
    shiftRegData[byteIndex] &= ~(1 << (6 - bitOffset));
  }
  
  // Blue bit
  if(b) {
    shiftRegData[byteIndex] |= (1 << (5 - bitOffset));
  } else {
    shiftRegData[byteIndex] &= ~(1 << (5 - bitOffset));
  }
  
  // Update row control (last byte)
  // For common anode, set the bit to 0 to activate the row
  shiftRegData[6] = ~(1 << y);
  
  // Shift out all data
  shiftOut48Bits(shiftRegData);
}

void clearMatrix() {
  // Initialize all shift register data to 1s (LEDs off for common anode)
  memset(shiftRegData, 0xFF, sizeof(shiftRegData));
  shiftOut48Bits(shiftRegData);
}