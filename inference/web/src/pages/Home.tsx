import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api, type PatientNights, type PatientSummary } from "../api";
import { Placeholder } from "../components/common";
import { ModelBar } from "../components/ModelBar";
import { formatDuration, patientPath, plural } from "../format";
import { useModel } from "../model";

const RATE_SCALE = 60;   // minutes per hour at a full bar

export function Home() {
  const query = useQuery({ queryKey: ["patients"], queryFn: api.patients });
  const { chosen } = useModel();
  const [filter, setFilter] = useState("");
  const search = useRef<HTMLInputElement>(null);

  const patients = query.data ?? [];
  const details = useQueries({
    queries: patients.map(p => ({
      queryKey: ["patient", p.id, chosen],
      queryFn: () => api.patient(p.id, chosen),
    })),
  });
  // The same per-patient query as the patient page, so opening a patient from here is instant.

  const lastSeen = useRef(new Map<string, PatientNights>());
  const rows = patients.map((patient, i) => {
    const fresh = details[i]?.data;
    if (fresh) lastSeen.current.set(patient.id, fresh);
    return { patient, nights: fresh ?? lastSeen.current.get(patient.id), stale: !fresh };
  });
  // After a model change a row keeps its last figures, dimmed, until the new model's arrive.

  useEffect(() => {
    document.title = "Patients · Apnea review";
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      if ((event.target as Element).closest("input, select, textarea")) return;
      event.preventDefault();
      search.current?.focus();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  if (query.isError) return <Placeholder title="Could not load patients">{query.error.message}</Placeholder>;
  if (query.isPending) return <Placeholder title="Loading…" />;
  if (!patients.length) {
    return (
      <Placeholder title="No patients yet">
        Copy each patient's recordings into <code>inference/data/patients/&lt;patient&gt;/</code>, one{" "}
        <code>.csv</code> file per night, then reload this page.
      </Placeholder>
    );
  }

  const term = filter.trim().toLowerCase();
  const shown = rows.filter(({ patient }) => patient.id.toLowerCase().includes(term));
  const nights = patients.reduce((sum, p) => sum + p.nights, 0);

  return (
    <>
      <h1 tabIndex={-1}>Patients</h1>
      <p className="subtitle">{plural(patients.length, "patient")} · {plural(nights, "night")}</p>
      <ModelBar>
        <input ref={search} className="search" type="search" placeholder="Search patients  ( / )"
               aria-label="Search patients" autoComplete="off" value={filter}
               onChange={event => setFilter(event.target.value)} />
      </ModelBar>
      {shown.length ? (
        <>
          <div className="list-head" aria-hidden="true"><span>Patient</span><span>Worst night</span></div>
          {shown.map(({ patient, nights, stale }) => (
            <PatientRow key={patient.id} patient={patient} nights={nights} stale={stale && nights !== undefined} />
          ))}
        </>
      ) : (
        <Placeholder title="No patient matches" />
      )}
    </>
  );
}

function PatientRow({ patient, nights, stale }: {
  patient: PatientSummary;
  nights?: PatientNights;
  stale: boolean;
}) {
  const recorded = nights?.nights.reduce((sum, n) => sum + n.duration_s, 0);
  const analysed = nights?.nights.filter(n => n.analysis) ?? [];
  const worst = analysed.reduce<(typeof analysed)[number] | null>(
    (top, n) => (!top || n.analysis!.apnea_minutes_per_hour > top.analysis!.apnea_minutes_per_hour ? n : top), null);

  return (
    <Link className="row-card row-link" to={patientPath(patient.id)} aria-busy={stale || undefined}>
      <div className="row-info">
        <h3 className="row-title">{patient.id}</h3>
        <p className="row-meta">
          {plural(patient.nights, "night")}
          {recorded !== undefined && ` · ${formatDuration(recorded)} recorded`}
          {analysed.length > 0 && ` · ${analysed.length} of ${patient.nights} analysed`}
        </p>
      </div>
      <div className="row-result"
           title={worst ? `Night ${worst.number}: ${worst.analysis!.apnea_minutes_per_hour.toFixed(1)} apnea minutes per hour` : undefined}>
        {!nights ? (
          <span className="muted">…</span>
        ) : !patient.nights ? (
          <span className="muted">No nights</span>
        ) : !worst ? (
          <span className="muted">Not analysed</span>
        ) : (
          <>
            <span className="rate-bar" aria-hidden="true">
              <span style={{ width: `${Math.min(100, (100 * worst.analysis!.apnea_minutes_per_hour) / RATE_SCALE)}%` }} />
            </span>
            <span className="rate">
              {worst.analysis!.apnea_minutes_per_hour.toFixed(1)}<span className="rate-unit"> min/h</span>
            </span>
          </>
        )}
      </div>
      <span className="chevron" aria-hidden="true">›</span>
    </Link>
  );
}
// "Worst night" is the analysed night with the most apnea minutes per hour, for the selected model.
