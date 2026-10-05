export const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function formatDuration(seconds: number): string {
  const minutes = Math.round(seconds / 60);
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return h ? `${h} h ${String(m).padStart(2, "0")} min` : `${m} min`;
}

export const formatClock = (minute: number) => `${Math.floor(minute / 60)}:${String(minute % 60).padStart(2, "0")}`;
// Minutes since the start of the recording, as h:mm.

export function formatSize(bytes: number): string {
  if (bytes < 1e3) return `${bytes} B`;
  if (bytes < 1e6) return `${Math.round(bytes / 1e3)} kB`;
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

export const patientPath = (patient: string) => `/patient/${encodeURIComponent(patient)}`;
export const nightPath = (patient: string, file: string) => `${patientPath(patient)}/night/${encodeURIComponent(file)}`;

export interface Period {
  start: number;
  length: number;
}

export function apneaPeriods(apnea: boolean[]): Period[] {
  const periods: Period[] = [];
  let start: number | null = null;
  apnea.forEach((flagged, i) => {
    if (flagged && start === null) start = i;
    if (!flagged && start !== null) {
      periods.push({ start, length: i - start });
      start = null;
    }
  });
  if (start !== null) periods.push({ start, length: apnea.length - start });
  return periods;
}
// Runs of consecutive flagged minutes.
