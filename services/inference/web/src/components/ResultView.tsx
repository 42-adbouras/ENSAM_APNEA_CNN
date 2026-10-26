import type { Analysis } from "../api";
import { apneaPeriods, formatClock, plural, type Period } from "../format";
import { Stat } from "./common";
import { NightChart } from "./NightChart";

function downloadCsv(r: Analysis, patient: string, file: string) {
  const lines = ["minute,start,probability,probability_smoothed,apnea"];
  r.prob.forEach((p, i) => lines.push([i + 1, formatClock(i), p, r.prob_smoothed[i], r.apnea[i] ? 1 : 0].join(",")));
  const url = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${patient}_${file.replace(/\.[^.]+$/, "")}_${r.model.replace(/\.tflite$/, "")}.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
// One row per minute: its start as h:mm, the raw and smoothed probabilities, and the decision.

function PeriodsTable({ periods }: { periods: Period[] }) {
  return (
    <details className="periods">
      <summary>List the {plural(periods.length, "flagged period")}</summary>
      <table>
        <thead>
          <tr><th>Start</th><th>End</th><th>Length</th></tr>
        </thead>
        <tbody>
          {periods.map(p => (
            <tr key={p.start}>
              <td>{formatClock(p.start)}</td>
              <td>{formatClock(p.start + p.length)}</td>
              <td>{p.length} min</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

export function ResultView({ r, patient, file }: { r: Analysis; patient: string; file: string }) {
  const periods = apneaPeriods(r.apnea);
  const longest = Math.max(0, ...periods.map(p => p.length));

  return (
    <div className="result">
      <dl className="stats">
        <Stat label="Apnea minutes" value={r.summary.apnea_minutes} sub={`of ${r.minutes}`} />
        <Stat label="Minutes per hour" value={r.summary.apnea_minutes_per_hour.toFixed(1)} />
        <Stat label="Flagged periods" value={periods.length} />
        <Stat label="Longest period" value={periods.length ? `${longest} min` : "None"} />
      </dl>
      <ul className="legend">
        <li><span className="key-apnea" />Flagged as apnea</li>
        <li><span className="key-line" />Apnea probability, {r.smooth_window}-minute average</li>
        <li><span className="key-threshold" />Threshold {r.threshold.toFixed(2)}</li>
      </ul>
      <NightChart key={`${r.model}/${r.night}`} r={r} />
      {periods.length > 0 && <PeriodsTable periods={periods} />}
      <div className="result-actions">
        <button type="button" className="btn" onClick={() => downloadCsv(r, patient, file)}>Download CSV</button>
        <button type="button" className="btn" onClick={() => window.print()}>Print</button>
      </div>
    </div>
  );
}
