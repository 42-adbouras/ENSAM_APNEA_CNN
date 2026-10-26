export interface Model {
  name: string;
  size_bytes: number;
  threshold: number | null;
}

export interface Models {
  default: string;
  models: Model[];
}

export interface PatientSummary {
  id: number;
  name: string;
  nights: number;
}

export interface Summary {
  apnea_minutes: number;
  apnea_fraction: number;
  apnea_minutes_per_hour: number;
}

export interface NightSummary extends Summary {
  minutes: number;
}

export interface Night {
  file: string;
  number: number;
  samples: number;
  duration_s: number;
  size_bytes: number;
  analysis: NightSummary | null;
}

export interface PatientNights {
  id: number;
  name: string;
  model: string;
  sample_rate_hz: number;
  nights: Night[];
}

export interface Analysis {
  patient: string;
  night: string;
  model: string;
  sample_rate_hz: number;
  minutes: number;
  threshold: number;
  smooth_window: number;
  summary: Summary;
  prob: number[];
  prob_smoothed: number[];
  apnea: boolean[];
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail ?? `${response.status} ${response.statusText}`, response.status);
  }
  return response.json() as Promise<T>;
}
// Errors carry the server's "detail" message and the HTTP status.

const enc = encodeURIComponent;
const modelQuery = (model: string | null) => (model ? `?model=${enc(model)}` : "");
const analysisUrl = (patient: number, file: string, model: string) =>
  `/api/patients/${patient}/nights/${enc(file)}/analysis${modelQuery(model)}`;

export const api = {
  patients: () => getJSON<PatientSummary[]>("/api/patients"),
  models: () => getJSON<Models>("/api/models"),
  patient: (patient: number, model: string | null) =>
    getJSON<PatientNights>(`/api/patients/${patient}${modelQuery(model)}`),

  async keptAnalysis(patient: number, file: string, model: string): Promise<Analysis | null> {
    try {
      return await getJSON<Analysis>(analysisUrl(patient, file, model));
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) return null;
      throw error;
    }
  },
  // The stored result for this model, or null when the night has not been analysed with it.

  analyse: (patient: number, file: string, model: string) =>
    getJSON<Analysis>(analysisUrl(patient, file, model), { method: "POST" }),

  createPatient: (name: string) =>
    getJSON<{ id: number; name: string }>("/api/patients", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),

  uploadNight(patient: number, file: File) {
    const form = new FormData();
    form.append("file", file);
    form.append("modified_ms", String(file.lastModified));
    return getJSON<{ id: number; file: string }>(`/api/patients/${patient}/nights`, {
      method: "POST",
      body: form,
    });
  },
  // lastModified is the file's modification time on this computer; the server gives it to the
  // saved copy, which sets the night's place in the order.
};

export const summaryOf = (result: Analysis): NightSummary => ({ minutes: result.minutes, ...result.summary });
