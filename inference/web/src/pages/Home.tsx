import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type PatientNights, type PatientSummary } from "../api";
import { Avatar, Placeholder, Spinner } from "../components/common";
import { ModelBar } from "../components/ModelBar";
import { formatDuration, patientPath, plural } from "../format";
import { useModel } from "../model";

export function Home() {
  const query = useQuery({ queryKey: ["patients"], queryFn: api.patients });
  const { chosen } = useModel();
  const [filter, setFilter] = useState("");
  const [creating, setCreating] = useState(false);
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

  const term = filter.trim().toLowerCase();
  const shown = rows.filter(({ patient }) => patient.id.toLowerCase().includes(term));
  const nights = patients.reduce((sum, p) => sum + p.nights, 0);

  return (
    <>
      <div className="page-head">
        <div>
          <h1 tabIndex={-1}>Patients</h1>
          <p className="subtitle">{plural(patients.length, "patient")} · {plural(nights, "night")}</p>
        </div>
        {!creating && (
          <button type="button" className="btn" onClick={() => setCreating(true)}>Create patient</button>
        )}
      </div>
      {creating && <NewPatientForm onClose={() => setCreating(false)} />}
      {!patients.length ? (
        <Placeholder title="No patients yet">
          Start with <strong>Create patient</strong>, then upload the patient's nights.
        </Placeholder>
      ) : (
        <>
          <ModelBar>
            <input ref={search} className="search" type="search" placeholder="Search patients  ( / )"
                   aria-label="Search patients" autoComplete="off" value={filter}
                   onChange={event => setFilter(event.target.value)} />
          </ModelBar>
          {shown.length ? (
            shown.map(({ patient, nights, stale }) => (
              <PatientRow key={patient.id} patient={patient} nights={nights} stale={stale && nights !== undefined} />
            ))
          ) : (
            <Placeholder title="No patient matches" />
          )}
        </>
      )}
    </>
  );
}

function NewPatientForm({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const create = useMutation({
    mutationFn: api.createPatient,
    onSuccess: async ({ id }) => {
      await queryClient.invalidateQueries({ queryKey: ["patients"] });
      navigate(patientPath(id));
    },
  });

  return (
    <form className="inline-form" onSubmit={event => { event.preventDefault(); create.mutate(name.trim()); }}>
      <label htmlFor="new-patient">Patient name</label>
      <input id="new-patient" className="search" autoFocus autoComplete="off" spellCheck={false}
             placeholder="e.g. patient_4" value={name}
             onChange={event => { setName(event.target.value); create.reset(); }}
             onKeyDown={event => { if (event.key === "Escape") onClose(); }} />
      <button type="submit" className="btn btn-primary" disabled={!name.trim() || create.isPending}>
        {create.isPending ? <><Spinner />Creating…</> : "Create"}
      </button>
      <button type="button" className="btn btn-ghost" onClick={onClose}>Cancel</button>
      <p className={create.isError ? "form-note error" : "form-note"}>
        {create.isError ? create.error.message : "Letters, digits, spaces, _ and -"}
      </p>
    </form>
  );
}
// Creating the patient opens its page, where its nights can be uploaded.

function PatientRow({ patient, nights, stale }: {
  patient: PatientSummary;
  nights?: PatientNights;
  stale: boolean;
}) {
  const recorded = nights?.nights.reduce((sum, n) => sum + n.duration_s, 0);
  const analysed = nights?.nights.filter(n => n.analysis).length;

  return (
    <Link className="row-card row-link" to={patientPath(patient.id)} aria-busy={stale || undefined}>
      <div className="row-info row-person">
        <Avatar />
        <div>
          <h3 className="row-title">{patient.id}</h3>
          <p className="row-meta">
            {plural(patient.nights, "night")}
            {recorded !== undefined && ` · ${formatDuration(recorded)} recorded`}
            {analysed !== undefined && patient.nights > 0 && ` · ${analysed} of ${patient.nights} analysed`}
          </p>
        </div>
      </div>
      <span className="chevron" aria-hidden="true">›</span>
    </Link>
  );
}
