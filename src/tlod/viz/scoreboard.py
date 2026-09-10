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

            age = (time.perf_counter() - self._result_at) if self._result else 0.0
            return {
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
}

function apply(d) {
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
    });
}
poll();
setInterval(poll, POLL);
</script>
""").replace("__POLL__", str(POLL_MS)).encode()
