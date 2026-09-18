"""The score, big enough to read from the other side of the table.

`--preview` answers "what is it looking at?" and the HUD in the corner of
that stream answers "what does it think?". Neither answers the question
the person with their hand on the table is actually asking, which is "did
I just get hit, and what is the score?". They are reading a 14 px HUD
line, in a stream that is deliberately throttled to 8 fps, while trying
to watch an arm. The verdict arrives too late and too small to be part of
the game.

So this is a second, much cheaper page: no camera frames, no JPEG, no
arm reads. The score is always on screen, and the instant a round
resolves the whole page flashes the result word in a colour that says who
won before the word has been read -- red when the robot scored, green
when the human did. That is the piece a person can take in with their
eyes on the arm rather than on the screen.

Two things shape the implementation:

The control loop runs at 100 Hz on an Orange Pi and must not be starved,
so nothing here runs on it. The game object is only ever *read*, from the
HTTP request thread, and the read is a handful of attribute loads. There
is no rendering, no encoding, and no extra thread.

`last_result` is cleared at the end of `settle`, roughly 600 ms after the
round resolves and unpredictably later if a flourish is still playing. A
browser polling for a truthy `last_result` would therefore see a flash of
random length, or miss it entirely, so the transition is latched here
instead: the round counter and the word are captured the first time they
are seen changed, and they stay captured until the next round. The
browser flashes on the round *number* changing, which also means a poll
lost to a hiccup costs nothing -- the next one still carries the result.
"""

from __future__ import annotations

import json
import threading
import time

# How often the browser asks. 200 ms is the slack between "the flash
# starts late enough to feel disconnected from the slap" and "the board
# is answering HTTP requests all day"; the response is ~120 bytes and
# costs an attribute read, so the cost is the request, not the work.
POLL_MS = 200

# How long the flash holds before it decays back to the score. Sized
# against `settle`, which sits still for at least 600 ms after a round:
# the flash should be over, and the score readable again, before the arm
# is moving for the next one.
FLASH_MS = 1100


class Scoreboard:
    """A read-only view of a game's score, with the result edge latched.

    Duck-typed rather than tied to HandSlapGame: it wants `score` with
    robot/human/rounds and a `last_result` string. Anything else it reads
    is optional and degrades to blank, because a scoreboard that raises
    is worse than a scoreboard that says less.
    """

    def __init__(self, policy) -> None:
        self.policy = policy
        self._lock = threading.Lock()
        self._result = ""
        self._result_round = 0
        self._result_at = 0.0
        self._flourish = ""
        self._flourish_count = 0
        self._flourish_at = 0.0
        self._seen: tuple[int, str] | None = None

    # -- state -------------------------------------------------------------
    def snapshot(self) -> dict:
        """Everything the page needs, and the latch step, in one call.

        Latching here rather than on a timer thread means the edge is
        only caught while somebody is watching, which is exactly when it
        matters and never costs anything when it does not. A round lasts
        seconds and the browser polls five times a second, so a result
        cannot slip between two polls of an open page.
        """
        policy = self.policy
        score = getattr(policy, "score", None)
        robot = int(getattr(score, "robot", 0) or 0)
        human = int(getattr(score, "human", 0) or 0)
        rounds = int(getattr(score, "rounds", 0) or 0)
        result = str(getattr(policy, "last_result", "") or "")

        with self._lock:
            # The pairing is the key, not the word alone: two hits in a
            # row are two flashes, and a word that reappears without the
            # counter moving is the same round still being displayed.
            #
            # `rounds` is incremented one line before `last_result` is
            # assigned, so a poll landing between them sees a new round
            # with a stale word. Requiring a non-empty result means that
            # poll simply latches nothing and the next one, 200 ms later,
            # gets it right -- rather than latching the previous round's
            # verdict under the new round's number.
            key = (rounds, result)
            if result and key != self._seen:
                self._seen = key
                self._result = result
                self._result_round = rounds
                self._result_at = time.perf_counter()

            # The gesture is its own edge. It starts about a second after
            # the verdict, so keying it off the round number would fire
            # the sound while the arm is still retracting; the counter the
            # game bumps when it actually starts performing is the honest
            # signal, and it also distinguishes two gloats in a row.
            count = int(getattr(policy, "flourishes", 0) or 0)
            if count != self._flourish_count:
                self._flourish_count = count
                self._flourish = str(getattr(policy, "last_flourish", "") or "")
                self._flourish_at = time.perf_counter()

            age = (time.perf_counter() - self._result_at) if self._result else 0.0
            return {
                "flourish": self._flourish,
                "flourish_n": self._flourish_count,
                "robot": robot,
                "human": human,
                "rounds": rounds,
                "result": self._result,
                "result_round": self._result_round,
                "age_ms": int(age * 1000),
                "state": str(getattr(policy, "state", "") or ""),
                "running": bool(getattr(policy, "running", True)),
                "flash_ms": FLASH_MS,
            }

    # -- serving -----------------------------------------------------------
    def json_route(self) -> tuple[str, bytes]:
        return "application/json", json.dumps(self.snapshot()).encode()

    def page_route(self) -> tuple[str, bytes]:
        return "text/html; charset=utf-8", PAGE


def serve(policy, port: int, server=None):
    """Publish `policy`'s score on `port`. Returns the server, or None.

    Takes an existing PreviewServer when there is one so that a run with
    both `--preview` and `--scoreboard` on the same port shares a single
    socket; otherwise it starts one of its own, with the scoreboard on
    `/` because there are no frames behind the usual index page.
    """
    if not port:
        return None
    from tlod.vision.preview import PreviewServer

    board = Scoreboard(policy)
    own = server is None
    if own:
        server = PreviewServer(port=port, label="scoreboard")
    server.add_route("/", board.page_route)
    server.add_route("/index.html", board.page_route)
    server.add_route("/score.json", board.json_route)
    if own:
        server.start()
    return server


# The page is one file with no dependencies on purpose. The board may
# have no route to a CDN, the laptop watching it may be on a phone
# hotspot, and a scoreboard that spends two seconds resolving fonts has
# already missed the round it was meant to show.
PAGE = ("""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>tlod scoreboard</title>
<style>
  :root { --bg:#0d0f12; --dim:#5c6470; --ink:#e8ecf1;
          --robot:#e2564a; --human:#4fc47f; }
  * { box-sizing:border-box; margin:0; }
  html,body { height:100%; }
  body { background:var(--bg); color:var(--ink); overflow:hidden;
         font:16px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif;
         display:flex; align-items:center; justify-content:center; }
  #score { display:flex; flex-direction:column; align-items:center;
           gap:2vh; transition:opacity .25s ease; }
  #tally { display:flex; align-items:center; gap:4vw; }
  .side { display:flex; flex-direction:column; align-items:center; gap:.6vh; }
  .who { font-size:2.6vh; letter-spacing:.35em; text-indent:.35em;
         text-transform:uppercase; color:var(--dim); }
  .num { font-size:22vh; font-weight:700; font-variant-numeric:tabular-nums;
         line-height:.9; }
  #robot .num { color:var(--robot); }
  #human .num { color:var(--human); }
  .dash { font-size:9vh; color:var(--dim); }
  #rounds { font-size:2.4vh; letter-spacing:.25em; text-indent:.25em;
            text-transform:uppercase; color:var(--dim); }
  /* The flash is a sibling that covers everything, not a body
     background: the score has to still be underneath it when it fades,
     so the eye lands back on the new number rather than on an empty
     screen that then fills in. */
  #flash { position:fixed; inset:0; display:flex; align-items:center;
           justify-content:center; opacity:0; pointer-events:none; }
  #flash.on { animation:flash var(--flash,1100ms) ease-out forwards; }
  #word { font-size:19vh; font-weight:800; letter-spacing:.06em;
          color:#0d0f12; }
  @keyframes flash {
    0%   { opacity:1; }
    55%  { opacity:1; }
    100% { opacity:0; }
  }
  #stale { position:fixed; bottom:2.5vh; left:0; right:0; text-align:center;
           font-size:2vh; letter-spacing:.2em; text-transform:uppercase;
           color:var(--dim); opacity:0; transition:opacity .4s ease; }
  body.stale #score { opacity:.25; }
  body.stale #stale { opacity:1; }
  /* Deliberately quiet and always visible. A browser will not make a
     sound until the page has been clicked, so a scoreboard that simply
     stayed silent would read as broken; this says which state it is in
     and how to change it, without competing with the score. */
  #sound { position:fixed; right:2vw; top:2vh; font-size:1.6vh;
           letter-spacing:.08em; text-transform:uppercase; color:var(--dim);
           cursor:pointer; user-select:none; }
  #sound.on { color:var(--human); }
  @media (prefers-reduced-motion:reduce) {
    #flash.on { animation-duration:400ms; }
  }
</style>
<div id=score>
  <div id=tally>
    <div class="side" id=robot><div class=who>robot</div><div class=num>0</div></div>
    <div class=dash>&ndash;</div>
    <div class="side" id=human><div class=who>you</div><div class=num>0</div></div>
  </div>
  <div id=rounds>round 0</div>
</div>
<div id=flash><div id=word></div></div>
<div id=stale>lost the board</div>
<div id=sound>sound off &middot; click or press S</div>
<script>
// Who the result favours, which is the thing to read first: red is a
// point for the arm, green is a point for the hand. The word is there to
// say which kind of point it was.
var COLOUR = {HIT:"#e2564a", FLINCH:"#e0a33c", DODGED:"#4fc47f", HELD:"#3fb0c8"};
var POLL = __POLL__;
var seen = null;                  // last round number we have flashed for
var elRobot = document.querySelector("#robot .num");
var elHuman = document.querySelector("#human .num");
var elRounds = document.getElementById("rounds");
var elFlash = document.getElementById("flash");
var elWord = document.getElementById("word");

function flash(word, ms) {
  elWord.textContent = word;
  elFlash.style.background = COLOUR[word] || "#8b93a0";
  elFlash.style.setProperty("--flash", ms + "ms");
  // Restarting a CSS animation needs the class off, a reflow, and the
  // class on again; without the reflow a second result inside one flash
  // does nothing at all, which is exactly the case that matters.
  elFlash.classList.remove("on");
  void elFlash.offsetWidth;
  elFlash.classList.add("on");
  sound(word);
}

// Synthesised rather than sampled. Four short files would have to be
// served from somewhere, and the whole point of this page is that it has
// no dependencies and works on a board with no route off the LAN -- so
// the slap is a noise burst and the rest are oscillators, which cost
// nothing to ship and never 404.
var actx = null, wantSound = false, noise = null;

function audio() {
  if (actx) { return actx; }
  var C = window.AudioContext || window.webkitAudioContext;
  if (!C) { return null; }
  actx = new C();
  // One second of white noise, reused for every slap. Building it per
  // hit would allocate 44100 floats on the same tick as the flash.
  noise = actx.createBuffer(1, actx.sampleRate, actx.sampleRate);
  var d = noise.getChannelData(0);
  for (var i = 0; i < d.length; i++) { d[i] = Math.random() * 2 - 1; }
  return actx;
}

function env(node, peak, attack, decay) {
  var g = actx.createGain();
  var t = actx.currentTime;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(peak, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay);
  node.connect(g);
  g.connect(actx.destination);
  node.start(t);
  node.stop(t + attack + decay + 0.02);
  return g;
}

function tone(type, from, to, peak, attack, decay) {
  var o = actx.createOscillator();
  o.type = type;
  var t = actx.currentTime;
  o.frequency.setValueAtTime(from, t);
  if (to !== from) { o.frequency.exponentialRampToValueAtTime(to, t + attack + decay); }
  env(o, peak, attack, decay);
}

function slap() {
  // A real slap is broadband and over in about a tenth of a second: a
  // filtered noise crack for the skin, a low sine for the weight behind
  // it. A pure tone reads as a beep no matter how short it is.
  var src = actx.createBufferSource();
  src.buffer = noise;
  var bp = actx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 1600;
  bp.Q.value = 0.7;
  src.connect(bp);
  var g = actx.createGain();
  var t = actx.currentTime;
  g.gain.setValueAtTime(0.9, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.13);
  bp.connect(g);
  g.connect(actx.destination);
  src.start(t);
  src.stop(t + 0.15);
  tone("sine", 180, 70, 0.5, 0.005, 0.11);
}

// One noise per gesture, so the performance is audible from wherever the
// player is actually looking, which is at their own hand. These are
// deliberately smaller than the verdict sounds: the verdict is the news
// and the flourish is the robot mugging about it, so a gloat that drowned
// out the slap would have the emphasis backwards.
function gesture(name) {
  if (!wantSound || !audio()) { return; }
  if (actx.state === "suspended") { actx.resume(); }
  // Three bites, and the arm is showing off: a rising arpeggio.
  if (name === "spin") { arp([440, 660, 880], 70, "triangle", 0.13); }
  // The shimmy is the wrist rolling and the jaw snapping; three clicks.
  else if (name === "shimmy") { arp([700, 700, 700], 110, "square", 0.09); }
  else if (name === "wag") { arp([500, 380], 110, "triangle", 0.11); }
  // Sulking, so both of these fall.
  else if (name === "nod") { arp([420, 300], 150, "sine", 0.13); }
  else if (name === "bob") { arp([320, 240], 150, "sine", 0.13); }
  // The jaw itself: three low clacks rather than a tune.
  else if (name === "chomp") { arp([160, 150, 140], 150, "square", 0.16); }
  else if (name === "jig") { arp([620, 780, 620, 780], 60, "square", 0.08); }
}

function arp(freqs, gap, type, peak) {
  freqs.forEach(function (f, i) {
    setTimeout(function () {
      if (actx) { tone(type, f, f, peak, 0.006, 0.09); }
    }, i * gap);
  });
}

function sound(word) {
  if (!wantSound || !audio()) { return; }
  if (actx.state === "suspended") { actx.resume(); }
  if (word === "HIT") { slap(); }
  // The arm bluffed and the hand moved: a short comic drop, because a
  // flinch is the one result that is the player's own fault.
  else if (word === "FLINCH") { tone("sawtooth", 520, 150, 0.22, 0.01, 0.26); }
  // Both of these are points for the hand, so both go up rather than
  // down, and the two-note version is the one you earned by holding.
  else if (word === "DODGED") { tone("square", 520, 880, 0.16, 0.008, 0.14); }
  else if (word === "HELD") {
    tone("sine", 660, 660, 0.2, 0.01, 0.16);
    setTimeout(function () { if (actx) { tone("sine", 990, 990, 0.2, 0.01, 0.3); } }, 130);
  }
}

var elSound = document.getElementById("sound");

function setSound(on) {
  wantSound = on;
  elSound.textContent = on ? "sound on \u00b7 press S to mute"
                           : "sound off \u00b7 click or press S";
  elSound.classList.toggle("on", on);
  try { localStorage.setItem("tlod.sound", on ? "1" : "0"); } catch (e) {}
  // Resuming has to happen inside the gesture that turned it on, or the
  // context stays suspended and the first few results are silent.
  if (on && audio() && actx.state === "suspended") { actx.resume(); }
}

// Default off. A page that starts making noise the moment it is opened
// is a page someone closes, and the browser would refuse anyway until it
// had been clicked.
var stored = null;
try { stored = localStorage.getItem("tlod.sound"); } catch (e) {}
setSound(stored === "1");

document.addEventListener("click", function () { setSound(!wantSound); });
document.addEventListener("keydown", function (e) {
  if (e.key === "s" || e.key === "S") { setSound(!wantSound); }
});

var seenGesture = null;           // last flourish counter we made a noise for

function apply(d) {
  applyGesture(d);
  elRobot.textContent = d.robot;
  elHuman.textContent = d.human;
  elRounds.textContent = "round " + d.rounds + (d.running ? "" : "  \\u00b7  paused");
  document.body.classList.remove("stale");
  if (!d.result) { return; }
  if (seen === null) {
    // First answer after a load or a reconnect. Flashing every past
    // round would be noise, but a page opened in the middle of one
    // should still show it, so replay only what is still on screen.
    seen = d.result_round;
    if (d.age_ms < d.flash_ms) { flash(d.result, d.flash_ms - d.age_ms); }
    return;
  }
  if (d.result_round !== seen) {
    seen = d.result_round;
    flash(d.result, d.flash_ms);
  }
}

function applyGesture(d) {
  if (!d.flourish_n) { return; }
  if (seenGesture === null) { seenGesture = d.flourish_n; return; }
  if (d.flourish_n !== seenGesture) {
    seenGesture = d.flourish_n;
    gesture(d.flourish);
  }
}

function poll() {
  fetch("score.json", {cache: "no-store"})
    .then(function (r) { return r.json(); })
    .then(apply)
    .catch(function () {
      // The run ended, or the board dropped off the network. Say so:
      // a score frozen at 3-2 looks identical to a game still being
      // played badly. `seen` is cleared so that the round in progress
      // when the link comes back still gets its flash.
      document.body.classList.add("stale");
      seen = null;
      seenGesture = null;
    });
}
poll();
setInterval(poll, POLL);
</script>
""").replace("__POLL__", str(POLL_MS)).encode()
