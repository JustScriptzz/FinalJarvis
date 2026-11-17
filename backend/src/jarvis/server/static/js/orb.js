/* ==========================================================================
   J.A.R.V.I.S. — the core orb
   --------------------------------------------------------------------------
   A canvas particle sphere, not a decorative loop. It is driven by real
   signals:

     state     one of idle | listening | processing | speaking | error
     level     0..1 loudness, from an AnalyserNode on the microphone or the
               speech output, so the sphere actually moves when you talk and
               when JARVIS talks back.

   No dependencies, no build step. It degrades to a static sphere when
   prefers-reduced-motion is set, and pauses when the tab is hidden so a
   background tab costs nothing.
   ========================================================================== */
(function () {
  "use strict";

  const TAU = Math.PI * 2;
  const POINTS = 1100;

  const PALETTE = {
    idle: { core: [126, 226, 245], shell: [96, 170, 200] },
    listening: { core: [90, 217, 160], shell: [70, 180, 140] },
    processing: { core: [126, 226, 245], shell: [130, 190, 230] },
    speaking: { core: [245, 189, 119], shell: [220, 160, 110] },
    error: { core: [255, 122, 134], shell: [200, 100, 110] },
  };

  // Fibonacci sphere: even coverage without clustering at the poles.
  function buildSphere(count) {
    const points = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let index = 0; index < count; index += 1) {
      const y = 1 - (index / (count - 1)) * 2;
      const radius = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = golden * index;
      points.push({
        x: Math.cos(theta) * radius,
        y,
        z: Math.sin(theta) * radius,
        // Stable per-point jitter so the surface never looks machined.
        seed: Math.random() * TAU,
        depth: 0.6 + Math.random() * 0.8,
      });
    }
    return points;
  }

  class Orb {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.points = buildSphere(POINTS);
      this.state = "idle";
      this.target = 0; // where `level` is heading
      this.level = 0; // smoothed loudness actually drawn
      this.levelSource = null; // pull-based loudness, set by bindLevel
      this.measuring = false; // true when the level source reports real audio
      this.spin = 0;
      this.tilt = 0;
      this.time = 0;
      this.visible = true;
      this.reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
      this.frame = null;
      this.last = 0;
      this.w = 0;
      this.h = 0;
      this.dpr = 1;

      this._resize = this._resize.bind(this);
      this._tick = this._tick.bind(this);

      window.addEventListener("resize", this._resize, { passive: true });
      // A canvas inside a hidden panel measures zero once and would stay blank
      // forever; observing the element catches the moment it is shown.
      if (window.ResizeObserver) {
        this.observer = new ResizeObserver(this._resize);
        this.observer.observe(canvas);
      }
      document.addEventListener("visibilitychange", () => {
        this.visible = !document.hidden;
        if (this.visible && this.frame === null) this.start();
        else if (!this.visible) this.stop();
      });
      this._resize();
      if (this.reduced) {
        this.level = 0.18;
        this._draw();
      } else {
        this.start();
      }
    }

    setState(state) {
      if (!PALETTE[state]) state = "idle";
      this.state = state;
      if (this.reduced) this._draw();
    }

    /** Feed a 0..1 loudness reading. Safe to call at any rate. */
    setLevel(value) {
      this.target = Math.max(0, Math.min(1, Number(value) || 0));
    }

    /**
     * Pull the level from a function on every frame instead of pushing it.
     *
     * A pull keeps the sphere in step with the audio graph even while the tab
     * throttles timers, and costs one extra call per frame. The function runs
     * inside try/catch: a closed AudioContext must never stop the orb.
     */
    bindLevel(fn) {
      this.levelSource = typeof fn === "function" ? fn : null;
    }

    start() {
      if (this.frame !== null || this.reduced) return;
      this.last = performance.now();
      this.frame = requestAnimationFrame(this._tick);
    }

    stop() {
      if (this.frame === null) return;
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }

    _resize() {
      const rect = this.canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      this.dpr = Math.min(2, window.devicePixelRatio || 1);
      this.w = rect.width;
      this.h = rect.height;
      this.canvas.width = Math.round(rect.width * this.dpr);
      this.canvas.height = Math.round(rect.height * this.dpr);
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      if (this.reduced) this._draw();
    }

    _tick(now) {
      const delta = Math.min(64, now - this.last) / 1000;
      this.last = now;
      this.time += delta;

      if (this.levelSource) {
        try {
          const measured = Number(this.levelSource()) || 0;
          // No audio graph (a suspended context, a host with no audio device, a
          // browser without Web Audio) means the measurement is flat while
          // JARVIS is plainly speaking. Rather than sit dead, the sphere falls
          // back to a quiet synthetic pulse for the speaking and listening
          // states, so presence never depends on optional hardware.
          this.measuring = measured > 0.001;
          if (this.measuring) {
            this.target = Math.max(0, Math.min(1, measured));
          } else if (this.state === "speaking" || this.state === "listening") {
            const pulse = 0.5 + Math.sin(this.time * (this.state === "speaking" ? 5.2 : 3.4)) * 0.5;
            this.target = 0.18 + pulse * 0.2;
          } else {
            this.target = 0;
          }
        } catch {
          this.levelSource = null;
        }
      }

      // Ease toward the target so the sphere never snaps between syllables.
      const rate = this.target > this.level ? 12 : 3.5;
      this.level += (this.target - this.level) * Math.min(1, delta * rate);

      // Each state has its own tempo; listening breathes faster than idle.
      const spinRate = this.state === "processing" ? 0.55 : 0.16;
      const breathe = this.state === "processing" ? 0.5 : 0.22;
      this.spin = (this.spin + delta * spinRate) % TAU;
      this.tilt = Math.sin(this.time * (this.state === "listening" ? 1.1 : 0.36)) * breathe;

      this._draw();
      this.frame = requestAnimationFrame(this._tick);
    }

    _draw() {
      const { ctx, w, h } = this;
      if (!w || !h) return;
      const paint = PALETTE[this.state] || PALETTE.idle;
      const cx = w / 2;
      const cy = h / 2;
      const base = Math.min(w, h) * 0.34;
      // Loudness expands the sphere and pushes points outward unevenly.
      const swell = 1 + this.level * 0.2;
      const radius = base * swell;

      ctx.clearRect(0, 0, w, h);

      // Halo behind the shell. The gradient has to fade to nothing inside the
      // canvas: stopping at the edge paints a visible square behind the sphere.
      const haloR = Math.min(radius * 1.9, Math.min(w, h) * 0.495);
      const halo = ctx.createRadialGradient(cx, cy, radius * 0.18, cx, cy, haloR);
      halo.addColorStop(0, `rgba(${paint.core[0]},${paint.core[1]},${paint.core[2]},${0.16 + this.level * 0.22})`);
      halo.addColorStop(0.55, `rgba(${paint.core[0]},${paint.core[1]},${paint.core[2]},0.07)`);
      halo.addColorStop(1, `rgba(${paint.core[0]},${paint.core[1]},${paint.core[2]},0)`);
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(cx, cy, haloR, 0, TAU);
      ctx.fill();

      const cosSpin = Math.cos(this.spin);
      const sinSpin = Math.sin(this.spin);
      const cosTilt = Math.cos(this.tilt);
      const sinTilt = Math.sin(this.tilt);
      const noise = this.level * 0.09;

      // Sort by depth so nearer points paint last.
      const projected = [];
      for (const point of this.points) {
        // Double rotation: spin around Y, then a gentle tilt.
        const x1 = point.x * cosSpin - point.z * sinSpin;
        const z1 = point.x * sinSpin + point.z * cosSpin;
        const y1 = point.y * cosTilt - z1 * sinTilt;
        const z2 = point.y * sinTilt + z1 * cosTilt;

        // Surface ripple. The phase depends on position in space, not on a
        // per-point random seed: adjacent points move together, so the sphere
        // breathes as one body instead of shedding scattered dots.
        const phase = this.time * 2.1 + y1 * 2.6 + x1 * 1.7;
        const ripple = 1 + Math.sin(phase) * noise * point.depth;
        const scale = ripple;
        const perspective = 1 / (1.9 - z2 * 0.55);
        projected.push({
          x: cx + x1 * radius * scale * perspective,
          y: cy + y1 * radius * scale * perspective,
          z: z2,
          depth: point.depth,
        });
      }
      projected.sort((a, b) => a.z - b.z);

      for (const p of projected) {
        // Front-facing points are brighter and larger. Loudness raises brightness
        // far more than blob size: at full scale, growing points would merge into
        // a solid mass and lose the sphere.
        const facing = (p.z + 1) / 2;
        const alpha = 0.06 + facing * (0.42 + this.level * 0.5);
        const size = 0.42 + facing * 1.15 * (0.8 + this.level * 0.28);
        ctx.fillStyle = `rgba(${paint.core[0]},${paint.core[1]},${paint.core[2]},${Math.min(1, alpha).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, TAU);
        ctx.fill();
      }

      // Outer shell line: a hairline, not a hard ring.
      ctx.strokeStyle = `rgba(${paint.shell[0]},${paint.shell[1]},${paint.shell[2]},${0.14 + this.level * 0.22})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 1.06, 0, TAU);
      ctx.stroke();

      // Inner core: the light source.
      const coreR = radius * (0.2 + this.level * 0.09);
      const core = ctx.createRadialGradient(cx, cy - coreR * 0.3, coreR * 0.1, cx, cy, coreR);
      core.addColorStop(0, "rgba(255,255,255,0.98)");
      core.addColorStop(0.35, `rgba(${paint.core[0]},${paint.core[1]},${paint.core[2]},0.92)`);
      core.addColorStop(1, `rgba(${paint.core[0]},${paint.core[1]},${paint.core[2]},0.05)`);
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(cx, cy, coreR, 0, TAU);
      ctx.fill();

      // Two orbit arcs, only while busy, to show work in progress.
      if (this.state === "processing" || this.state === "listening") {
        const orbit = radius * 1.22;
        for (let index = 0; index < 2; index += 1) {
          const start = this.time * (index ? -0.9 : 1.4) + index * 2.1;
          ctx.strokeStyle = `rgba(${paint.core[0]},${paint.core[1]},${paint.core[2]},${0.3 + this.level * 0.3})`;
          ctx.lineWidth = 1.1;
          ctx.beginPath();
          ctx.arc(cx, cy, orbit + index * 5, start, start + 0.9);
          ctx.stroke();
        }
      }
    }
  }

  window.JarvisOrb = Orb;
})();
