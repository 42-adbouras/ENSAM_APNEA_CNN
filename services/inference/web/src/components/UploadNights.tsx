import { useRef, useState, type ChangeEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { plural } from "../format";
import { Spinner } from "./common";

export interface UploadProgress {
  done: number;
  total: number;
}

export interface UploadReport {
  uploaded: string[];
  failed: { file: string; reason: string }[];
}

export function useNightUpload(patient: string) {
  const queryClient = useQueryClient();
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [report, setReport] = useState<UploadReport | null>(null);

  async function upload(files: File[]) {
    const csv = files.filter(f => f.name.toLowerCase().endsWith(".csv"));
    const result: UploadReport = {
      uploaded: [],
      failed: files.filter(f => !csv.includes(f)).map(f => ({ file: f.name, reason: "not a .csv file" })),
    };
    setReport(null);
    for (const [i, file] of csv.entries()) {
      setProgress({ done: i + 1, total: csv.length });
      try {
        await api.uploadNight(patient, file);
        result.uploaded.push(file.name);
        queryClient.invalidateQueries({ queryKey: ["patient", patient] });
      } catch (error) {
        result.failed.push({ file: file.name, reason: (error as Error).message });
      }
    }
    queryClient.invalidateQueries({ queryKey: ["patients"] });
    setProgress(null);
    setReport(result);
  }
  // Files not ending in .csv are left out before anything is sent. The others are sent one after
  // another; the patient's night list is refetched after each one that succeeds.

  return { progress, report, upload };
}

export function UploadButton({ progress, onFiles }: {
  progress: UploadProgress | null;
  onFiles: (files: File[]) => void;
}) {
  const input = useRef<HTMLInputElement>(null);

  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (files.length) onFiles(files);
  };
  // The input is emptied after each choice, so choosing the same file again still fires onChange.

  return (
    <>
      <input ref={input} type="file" accept=".csv" multiple hidden onChange={onChange} />
      <button type="button" className={progress ? "btn busy" : "btn"} disabled={progress !== null}
              onClick={() => input.current?.click()}>
        {progress ? <><Spinner />Uploading {progress.done} of {progress.total}…</> : "Upload nights"}
      </button>
    </>
  );
}

export function UploadResult({ report }: { report: UploadReport | null }) {
  if (!report) return null;
  return (
    <div className="upload-report" role="status">
      {report.uploaded.length > 0 && (
        <p className="note">{plural(report.uploaded.length, "night")} uploaded.</p>
      )}
      {report.failed.length > 0 && (
        <>
          <p className="note error">{plural(report.failed.length, "file")} not uploaded:</p>
          <ul>
            {report.failed.map((f, i) => (
              <li key={i}><code className="file">{f.file}</code>: {f.reason}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
