"""MicroPython firmware for the Raspberry Pi Pico sidecar.

The Pico cannot do vision -- it is roughly a thousand times short of hand
detection, and no firmware trick closes that. What it is uniquely good at
is the job vision is bad at: reacting to physical events in microseconds,
and doing so whether or not the host is healthy.

Two jobs matter, in this order.

1. HARDWARE E-STOP. A button wired to this board cuts servo power through
   a relay *directly*, in the interrupt handler, before it tells anyone.
   The host is informed afterwards as a courtesy. This is the only stop
   on the robot that still works when Python has hung, the USB cable has
   fallen out, or the control loop has deadlocked -- which is precisely
   when a machine that swings at hands needs stopping. Software e-stop is
   a convenience; this is the safety device.

2. IMPACT DETECTION. A piezo disc under the target pad timestamps a slap
   in microseconds. The camera cannot do this: at the moment of contact
   the arm is between the lens and the contact point, and 30 fps gives
   33 ms of granularity on an event that decides the round.

Everything else here -- LEDs, buzzer, score display -- is arcade-cabinet
polish that simply does not belong on the host's critical path.

  !! UNVERIFIED ON HARDWARE !!
  Written against the RP2040/RP2350 MicroPython API while the boards were
  still on order. The wire protocol is fixed and the host side
  (tlod.game.contact.SerialContactSensor) speaks it, but neither end has
  run on a real board. Bring-up order is in docs/deployment.md.

Wiring
------
  GP26 / ADC0   piezo disc, with a 1 MΩ bleed resistor to ground and two
                clamp diodes to 3V3 and GND. A piezo can output tens of
                volts on a sharp hit; without clamps it will destroy the
                ADC input.
  GP15          e-stop button to ground, internal pull-up
  GP14          servo power relay / MOSFET gate, active high = powered
  GP16,17,18    status LEDs (ready, hit, estop)
  GP19          passive buzzer

Protocol, newline-delimited ASCII both ways
-------------------------------------------
  ->host  READY <version>
  ->host  HIT <micros> <amplitude 0..1>
  ->host  ESTOP <micros>
  ->host  CLEAR <micros>          e-stop released
  ->host  PONG <micros>
  host->  PING
  host->  ARM                     re-enable servo power after an e-stop
  host->  BEEP <ms> <freq>
  host->  LED <ready> <hit> <estop>    each 0 or 1
"""

VERSION = "1"

try:
    from machine import ADC, PWM, Pin
    import utime as time
    import sys
    import select
    MICROPYTHON = True
except ImportError:  # pragma: no cover - lets the file be imported for linting
    MICROPYTHON = False

PIN_PIEZO = 26
PIN_ESTOP_BUTTON = 15
PIN_SERVO_POWER = 14
PIN_LED_READY, PIN_LED_HIT, PIN_LED_ESTOP = 16, 17, 18
PIN_BUZZER = 19

# Piezo threshold as a fraction of full scale. Set this by watching real
# readings with the pad tapped and the pad ignored: too low and footsteps
# on the table register as hits, too high and a glancing slap does not.
HIT_THRESHOLD = 0.18
# After a hit, ignore the ringing. A piezo oscillates for milliseconds
# after contact and would otherwise report one slap as a dozen.
HIT_DEBOUNCE_US = 120_000


class Sidecar:
    def __init__(self) -> None:
        self.piezo = ADC(Pin(PIN_PIEZO))
        self.button = Pin(PIN_ESTOP_BUTTON, Pin.IN, Pin.PULL_UP)
        self.servo_power = Pin(PIN_SERVO_POWER, Pin.OUT, value=1)
        self.led_ready = Pin(PIN_LED_READY, Pin.OUT, value=1)
        self.led_hit = Pin(PIN_LED_HIT, Pin.OUT, value=0)
        self.led_estop = Pin(PIN_LED_ESTOP, Pin.OUT, value=0)
        self.buzzer = PWM(Pin(PIN_BUZZER))
        self.buzzer.duty_u16(0)

        self.estopped = False
        self.last_hit_us = 0
        self.baseline = self._measure_baseline()

        # Cut power in the handler itself. Not by setting a flag for the
        # main loop to notice: if the main loop is what has gone wrong,
        # the flag is never read.
        self.button.irq(trigger=Pin.IRQ_FALLING, handler=self._on_estop)

    # -- safety ------------------------------------------------------------
    def _on_estop(self, pin) -> None:
        self.servo_power.value(0)
        self.estopped = True
        self.led_estop.value(1)
        self.led_ready.value(0)
        # Reporting comes after cutting power, deliberately.
        self._send("ESTOP", time.ticks_us())

    def arm(self) -> None:
        """Re-enable servo power. Refused while the button is still held."""
        if self.button.value() == 0:
            return
        self.servo_power.value(1)
        self.estopped = False
        self.led_estop.value(0)
        self.led_ready.value(1)
        self._send("CLEAR", time.ticks_us())

    # -- impact ------------------------------------------------------------
    def _measure_baseline(self) -> int:
        """Average the resting piezo reading, so the threshold is relative.

        A piezo sits at whatever its bias network puts it at, and that
        drifts with temperature. An absolute threshold would need
        retuning on a cold morning.
        """
        total = 0
        for _ in range(256):
            total += self.piezo.read_u16()
            time.sleep_us(50)
        return total // 256

    def poll_piezo(self) -> None:
        raw = self.piezo.read_u16()
        deviation = abs(raw - self.baseline) / 32768.0
        if deviation < HIT_THRESHOLD:
            return
        now = time.ticks_us()
        if time.ticks_diff(now, self.last_hit_us) < HIT_DEBOUNCE_US:
            return
        self.last_hit_us = now
        amplitude = deviation if deviation < 1.0 else 1.0
        self._send("HIT", now, "%.3f" % amplitude)
        self.led_hit.value(1)
        self._hit_led_off_at = time.ticks_add(now, 150_000)

    # -- io ----------------------------------------------------------------
    def _send(self, *parts) -> None:
        print(" ".join(str(p) for p in parts))

    def beep(self, ms: int, freq: int) -> None:
        self.buzzer.freq(max(50, min(freq, 5000)))
        self.buzzer.duty_u16(12000)
        self._buzzer_off_at = time.ticks_add(time.ticks_us(), ms * 1000)

    def handle(self, line: str) -> None:
        parts = line.strip().split()
        if not parts:
            return
        command = parts[0].upper()
        if command == "PING":
            self._send("PONG", time.ticks_us())
        elif command == "ARM":
            self.arm()
        elif command == "BEEP" and len(parts) >= 3:
            self.beep(int(parts[1]), int(parts[2]))
        elif command == "LED" and len(parts) >= 4:
            self.led_ready.value(int(parts[1]))
            self.led_hit.value(int(parts[2]))
            self.led_estop.value(int(parts[3]))

    def run(self) -> None:
        self._send("READY", VERSION)
        poller = select.poll()
        poller.register(sys.stdin, select.POLLIN)
        self._buzzer_off_at = 0
        self._hit_led_off_at = 0

        while True:
            # The piezo is polled first and unconditionally. Host chatter
            # must never delay impact detection, which is the one thing
            # here with a microsecond budget.
            if not self.estopped:
                self.poll_piezo()

            now = time.ticks_us()
            if self._buzzer_off_at and time.ticks_diff(now, self._buzzer_off_at) > 0:
                self.buzzer.duty_u16(0)
                self._buzzer_off_at = 0
            if self._hit_led_off_at and time.ticks_diff(now, self._hit_led_off_at) > 0:
                self.led_hit.value(0)
                self._hit_led_off_at = 0

            if poller.poll(0):
                self.handle(sys.stdin.readline())


if MICROPYTHON:
    Sidecar().run()
