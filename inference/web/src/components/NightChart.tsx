import { useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Analysis } from "../api";
import { apneaPeriods, formatClock } from "../format";

const M = { top: 2, right: 6, bottom: 22, left: 30 };
const STRIP_H = 10;
const GAP = 12;
const PLOT_H = 110;
const PLOT_TOP = M.top + STRIP_H + GAP;
const HEIGHT = PLOT_TOP + PLOT_H + M.bottom;
const TIP_ROOM = 200;   // px a tooltip needs to the right of the crosshair before it flips to the left

export function NightChart({ r }: { r: Analysis }) {
  const box = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(0);
  const [current, setCurrent] = useState<number | null>(null);

  useLayoutEffect(() => {
    const element = box.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setWidth(element.clientWidth));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  // The chart is drawn at its container's pixel width and redrawn when that width changes.

  const n = r.minutes;
  const plotW = Math.max(0, width - M.left - M.right);
  const xAt = (minute: number) => M.left + (minute * plotW) / n;
  const yAt = (p: number) => PLOT_TOP + (1 - p) * PLOT_H;
  const centre = (i: number) => xAt(i + 0.5);

  const periods = useMemo(() => apneaPeriods(r.apnea), [r]);
  const points = useMemo(
    () => r.prob_smoothed.map((p, i) => `${centre(i).toFixed(1)},${yAt(p).toFixed(1)}`).join("L"),
    [r, width],
  );
  // centre() and yAt() change only with the width and the number of minutes in r.

  const hours = Math.floor(n / 60);
  const step = plotW / (n / 60) < 36 ? 2 : 1;
  const hourTicks = Array.from({ length: Math.floor(hours / step) + 1 }, (_, i) => i * step);

  const show = (minute: number) => setCurrent(Math.max(0, Math.min(n - 1, minute)));

  const onPointerMove = (event: PointerEvent<SVGRectElement>) => {
    const left = svgRef.current?.getBoundingClientRect().left ?? 0;
    show(Math.floor(((event.clientX - left - M.left) / plotW) * n));
  };

  const onKeyDown = (event: KeyboardEvent<SVGSVGElement>) => {
    const moves: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1, Home: -n, End: n };
    if (!(event.key in moves)) return;
    event.preventDefault();
    show((current ?? 0) + moves[event.key] * (event.shiftKey ? 10 : 1));
  };
  // The pointer snaps to the nearest minute over the strip or the plot. Arrows move one minute,
  // Shift+arrow ten, Home and End to either end.

  const x = current === null ? 0 : centre(current);
  const flip = x + TIP_ROOM > width;

  return (
    <div className="chart" ref={box}>
      {width > 0 && (
        <svg ref={svgRef} width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} role="img" tabIndex={0}
             aria-label={`Apnea probability for each of ${n} minutes, ${r.summary.apnea_minutes} flagged. ` +
                         "Use the arrow keys to read minute by minute."}
             onKeyDown={onKeyDown} onFocus={() => show(current ?? 0)} onBlur={() => setCurrent(null)}>
          <rect className="frame" x={0.5} y={0.5} width={width - 1} height={HEIGHT - 1} rx={4} />

          <rect className="c-track" x={M.left} y={M.top} width={plotW} height={STRIP_H} rx={2} />
          {periods.map(p => (
            <rect key={p.start} className="c-apnea" x={xAt(p.start)} y={M.top}
                  width={Math.max(1, (p.length * plotW) / n)} height={STRIP_H} />
          ))}

          {[0, 0.5, 1].map(p => (
            <g key={p}>
              <line className="c-grid" x1={M.left} x2={M.left + plotW} y1={yAt(p)} y2={yAt(p)} />
              <text className="c-tick" x={M.left - 8} y={yAt(p)} textAnchor="end" dominantBaseline="middle">{p}</text>
            </g>
          ))}

          <path className="c-area" d={`M${centre(0)},${yAt(0)}L${points}L${centre(n - 1)},${yAt(0)}Z`} />
          <line className="c-threshold" x1={M.left} x2={M.left + plotW} y1={yAt(r.threshold)} y2={yAt(r.threshold)} />
          <path className="c-line" d={`M${points}`} />

          {hourTicks.map(h => (
            <text key={h} className="c-tick" x={xAt(h * 60)} y={HEIGHT - 6} textAnchor={h === 0 ? "start" : "middle"}>
              {h} h
            </text>
          ))}

          {current !== null && (
            <>
              <line className="c-cross" x1={x} x2={x} y1={M.top} y2={PLOT_TOP + PLOT_H} />
              <circle className="c-dot" r={4} cx={x} cy={yAt(r.prob_smoothed[current])} />
            </>
          )}
          <rect x={M.left} y={0} width={plotW} height={PLOT_TOP + PLOT_H} fill="transparent"
                onPointerMove={onPointerMove} onPointerLeave={() => setCurrent(null)} />
        </svg>
      )}

      {current !== null && (
        <div className="tip" style={{ left: flip ? x - 14 : x + 14, top: PLOT_TOP, transform: flip ? "translateX(-100%)" : undefined }}>
          <div className="row">
            <span className="key-line" />
            <strong>{r.prob_smoothed[current].toFixed(2)}</strong>probability
          </div>
          <div className="row">
            {r.apnea[current] && <span className="key-apnea" />}
            {r.apnea[current] ? "Flagged as apnea" : "Not flagged"}
          </div>
          <div className="muted">Minute {current + 1}, at {formatClock(current)}</div>
        </div>
      )}
    </div>
  );
}
