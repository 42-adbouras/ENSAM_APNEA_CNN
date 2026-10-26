import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, summaryOf, type Night, type PatientNights } from "../api";
import { Avatar, Crumbs, NightMeta, Placeholder, Spinner, Stat } from "../components/common";
import { ModelBar } from "../components/ModelBar";
import { UploadButton, UploadResult, useNightUpload } from "../components/UploadNights";
import { formatDuration, nightPath } from "../format";
import { useModel } from "../model";

const RATE_SCALE = 60;   // minutes per hour at a full bar

export function PatientPage() {
  const { patient = "" } = useParams();
  return <PatientView key={patient} patient={Number(patient)} />;
}
// key={patient} gives each patient a fresh view, which stops an "Analyse all" run of the previous one.

interface Running {
  done: number;
  total: number;
  file: string;
}

function PatientView({ patient }: { patient: number }) {
  const { chosen } = useModel();
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["patient", patient, chosen],
    queryFn: () => api.patient(patient, chosen),
    placeholderData: keepPreviousData,
  });
  // keepPreviousData: after a model change the rows stay on screen, dimmed, until the new ones arrive.

  const [running, setRunning] = useState<Running | null>(null);
  const [failures, setFailures] = useState<Record<string, string>>({});
  const upload = useNightUpload(patient);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    if (query.data) document.title = `${query.data.name} · Apnea review`;
  }, [query.data]);

  async function analyseAll() {
    if (!chosen || !query.data) return;
    const model = chosen;
    const pending = [...query.data.nights].reverse().filter(n => !n.analysis);
    // Newest first, the order of the list on screen.
    for (const [i, night] of pending.entries()) {
      if (!alive.current) return;
      setRunning({ done: i + 1, total: pending.length, file: night.file });
      try {
        const result = await api.analyse(patient, night.file, model);
        queryClient.setQueryData(["analysis", patient, night.file, model], result);
        queryClient.setQueryData<PatientNights>(["patient", patient, model], old => old && {
          ...old,
          nights: old.nights.map(n => (n.file === night.file ? { ...n, analysis: summaryOf(result) } : n)),
        });
      } catch (error) {
        setFailures(f => ({ ...f, [`${model}/${night.file}`]: (error as Error).message }));
      }
    }
    if (alive.current) setRunning(null);
  }
  // One night after another; each result goes straight into the query cache, so its row and its
  // night page update without another request.

  const nights = query.data?.nights;
  const done = nights?.filter(n => n.analysis).length ?? 0;
  const recorded = nights?.reduce((sum, n) => sum + n.duration_s, 0) ?? 0;

  return (
    <>
      <Crumbs items={[{ label: "Patients", to: "/" }, { label: query.data?.name ?? "" }]} />
      <div className="page-head">
        <div className="title-row">
          <Avatar large />
          <h1 tabIndex={-1}>{query.data?.name ?? (query.isError ? "Patient" : "Loading…")}</h1>
        </div>
        {!query.isError && <UploadButton progress={upload.progress} onFiles={upload.upload} />}
      </div>
      <UploadResult report={upload.report} />
      {query.isError ? (
        <Placeholder title="Could not open this patient">{query.error.message}</Placeholder>
      ) : !nights ? (
        <Placeholder title="Loading…" />
      ) : !nights.length ? (
        <Placeholder title="No nights yet">
          Add this patient's recordings with <strong>Upload nights</strong>, one <code>.csv</code> file
          per night.
        </Placeholder>
      ) : (
        <>
          <dl className="stats">
            <Stat label="Nights" value={nights.length} />
            <Stat label="Recorded" value={formatDuration(recorded)} />
            <Stat label="Analysed" value={`${done} of ${nights.length}`} />
          </dl>
          <ModelBar action={{
            label: done === nights.length ? "All nights analysed" : "Analyse all nights",
            busyText: running && `Analysing ${running.done} of ${running.total}…`,
            disabled: done === nights.length,
            run: analyseAll,
          }} />
          <div aria-busy={query.isPlaceholderData || undefined}>
            {[...nights].reverse().map(night => (
              <NightRow key={night.file} patient={patient} night={night}
                        busy={running?.file === night.file} failure={failures[`${chosen}/${night.file}`]} />
            ))}
            {/* The server lists nights oldest first (Night 1); the page shows the newest at the top. */}
          </div>
        </>
      )}
    </>
  );
}

function NightRow({ patient, night, busy, failure }: {
  patient: number;
  night: Night;
  busy: boolean;
  failure?: string;
}) {
  const analysis = night.analysis;
  const title = analysis ? `${analysis.apnea_minutes} of ${analysis.minutes} minutes flagged` : failure;

  return (
    <Link className="row-card row-link" to={nightPath(patient, night.file)}>
      <div className="row-info">
        <h3 className="row-title">Night {night.number}</h3>
        <NightMeta night={night} />
      </div>
      <div className="row-result" title={title}>
        {busy ? (
          <><span className="muted"><Spinner /></span><span className="muted">Analysing…</span></>
        ) : analysis ? (
          <>
            <span className="rate-bar" aria-hidden="true">
              <span style={{ width: `${Math.min(100, (100 * analysis.apnea_minutes_per_hour) / RATE_SCALE)}%` }} />
            </span>
            <span className="rate">
              {analysis.apnea_minutes_per_hour.toFixed(1)}<span className="rate-unit"> min/h</span>
            </span>
          </>
        ) : failure ? (
          <span className="error">Analysis failed</span>
        ) : (
          <span className="muted">Not analysed</span>
        )}
      </div>
      <span className="chevron" aria-hidden="true">›</span>
    </Link>
  );
}
