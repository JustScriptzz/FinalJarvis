/* ==========================================================================
   J.A.R.V.I.S. — audio meter
   --------------------------------------------------------------------------
   Metering for the two things that make sound: the microphone and JARVIS's
   own voice. The orb reads `level()` from here, so the sphere swells with the
   real loudness of speech instead of a decorative timer.

   The two paths stay separate on purpose. A microphone source must never be
   connected to the destination — that is a feedback loop through the speakers.
   So the mic gets its own analyser connected to nothing, and the speech output
   gets a second analyser on the element's route to the speakers.

   Everything is optional by design: a browser without Web Audio, a denied
   microphone or a muted tab all report a level of 0 and nothing throws.
   ========================================================================== */
(function () {
  "use strict";

  class AudioMeter {
    constructor() {
      this.ctx = null;
      this.mic = null; // analyser fed by the microphone, connected to nothing
      this.out = null; // analyser on the speech path to the speakers
      this.micBuffer = null;
      this.outBuffer = null;
      this.stream = null;
      this.source = null;
      this.sink = null;
    }

    /** Lazily create the context; browsers require a gesture first. */
    _ensure() {
      if (this.ctx) return this.ctx;
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      this.ctx = new Ctx();
      this.mic = this._analyser();
      this.out = this._analyser();
      this.micBuffer = new Uint8Array(this.mic.fftSize);
      this.outBuffer = new Uint8Array(this.out.fftSize);
      return this.ctx;
    }

    _analyser() {
      const node = this.ctx.createAnalyser();
      // 512 samples is about 11ms at 44.1kHz: responsive without flickering.
      node.fftSize = 512;
      node.smoothingTimeConstant = 0.7;
      return node;
    }

    get context() {
      return this._ensure();
    }

    /**
     * Ask the context to start, without ever blocking the caller.
     *
     * Browsers suspend a context until a gesture, and some environments (a
     * headless browser, a host with no audio device) never resolve the resume
     * promise at all. Metering is a nicety, so it must not be able to hold up
     * the microphone: the promise is started and abandoned, and `level()` keeps
     * returning 0 until the context is genuinely running.
     */
    resume() {
      const ctx = this._ensure();
      if (!ctx) return null;
      if (ctx.state === "suspended") {
        try {
          const pending = ctx.resume();
          if (pending && typeof pending.catch === "function") {
            pending.catch(() => {
              /* a blocked context simply leaves the orb still */
            });
          }
        } catch {
          /* an unusable context must not stop recording */
        }
      }
      return ctx;
    }

    /** True once audio is actually flowing through the graph. */
    get running() {
      return Boolean(this.ctx && this.ctx.state === "running");
    }

    /**
     * Route a microphone stream through its analyser.
     *
     * The recorder has already been handed this same stream. Stopping the
     * tracks here would end the recording, so `detachStream` is the only place
     * that releases them, and it is called after recording stops.
     */
    attachStream(stream) {
      const ctx = this._ensure();
      if (!ctx || !stream) return;
      this.releaseStream();
      this.stream = stream;
      try {
        this.source = ctx.createMediaStreamSource(stream);
        this.source.connect(this.mic);
      } catch {
        // A stream with no audio track cannot be measured; recording still works.
        this.source = null;
      }
    }

    /** Disconnect the meter without stopping the tracks the recorder owns. */
    releaseStream() {
      if (this.source) {
        try {
          this.source.disconnect();
        } catch {
          /* already gone */
        }
        this.source = null;
      }
      this.stream = null;
    }

    /** Route an <audio> element through the output analyser on its way out. */
    /**
     * Route an <audio> element through the output analyser on its way out.
     *
     * Returns whether the element is actually being routed. Once an element is
     * tapped, its audio reaches the speakers only through this graph, so a
     * context that is not running would silence it. Callers therefore check the
     * result and fall back to native playback rather than going mute.
     */
    attachElement(element) {
      const ctx = this._ensure();
      if (!ctx || !element) return false;
      // A suspended context cannot carry the element's audio to the speakers.
      if (ctx.state !== "running") return false;
      this.detachElement();
      try {
        this.sink = ctx.createMediaElementSource(element);
        // Element sources must reach the destination through the graph, so this
        // analyser sits inline: element -> analyser -> speakers.
        this.sink.connect(this.out);
        this.out.connect(ctx.destination);
        return true;
      } catch {
        // An element already tapped, or one that cannot be routed, plays normally.
        this.sink = null;
        return false;
      }
    }

    detachElement() {
      if (this.sink) {
        try {
          this.sink.disconnect();
        } catch {
          /* already gone */
        }
        this.sink = null;
      }
      if (this.out) {
        try {
          this.out.disconnect();
        } catch {
          /* already gone */
        }
      }
    }

    _rms(analyser, buffer) {
      if (!analyser || !buffer) return 0;
      analyser.getByteTimeDomainData(buffer);
      let sum = 0;
      for (let index = 0; index < buffer.length; index += 1) {
        const centred = (buffer[index] - 128) / 128;
        sum += centred * centred;
      }
      return Math.sqrt(sum / buffer.length);
    }

    /** Microphone loudness, 0..1. */
    micLevel() {
      const rms = this._rms(this.mic, this.micBuffer);
      // Speech RMS sits near 0.02-0.15; scale it into something the orb can use.
      return Math.min(1, Math.pow(rms * 4.2, 0.7));
    }

    /** JARVIS's own voice loudness, 0..1. */
    outputLevel() {
      const rms = this._rms(this.out, this.outBuffer);
      return Math.min(1, Math.pow(rms * 4.2, 0.7));
    }

    /** Whatever is loudest right now. */
    level() {
      return Math.max(this.micLevel(), this.outputLevel());
    }

    close() {
      this.releaseStream();
      this.detachElement();
      this.mic = null;
      this.out = null;
      this.micBuffer = null;
      this.outBuffer = null;
      if (this.ctx) {
        try {
          this.ctx.close();
        } catch {
          /* already closed */
        }
        this.ctx = null;
      }
    }
  }

  window.JarvisAudioMeter = AudioMeter;
})();
