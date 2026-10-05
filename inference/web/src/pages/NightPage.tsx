import { useEffect, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Night } from "../api";
import { Crumbs, NightMeta, Placeholder } from "../components/common";
import { ModelBar } from "../components/ModelBar";
import { ResultView } from "../components/ResultView";
import { nightPath, patientPath } from "../format";
import { useModel } from "../model";

export function NightPage() {
  const { patient = "", night = "" } = useParams();
  return <NightView key={`${patient}/${night}`} patient={patient} file={night} />;
}

function PagerLink({ patient, night, children }: { patient: string; night?: Night; children: ReactNode }) {
  if (!night) return <span className="btn btn-ghost" aria-disabled="true">{children}</span>;
  return <Link className="btn btn-ghost" to={nightPath(patient, night.file)}>{children}</Link>;
}

function NightView({ patient, file }: { patient: string; file: string }) {
  const { chosen } = useModel();
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const nightsQuery = useQuery({
    queryKey: ["patient", patient, chosen],
    queryFn: () => api.patient(patient, chosen),
    placeholderData: keepPreviousData,
  });
  const nights = nightsQuery.data?.nights;
  const index = nights ? nights.findIndex(n => n.file === file) : -1;
  const night = index >= 0 ? nights![index] : undefined;
  const prev = index > 0 ? nights![index - 1] : undefined;
  const next = nights && index >= 0 && index < nights.length - 1 ? nights[index + 1] : undefined;

  const kept = useQuery({
    queryKey: ["analysis", patient, file, chosen],
    queryFn: () => api.keptAnalysis(patient, file, chosen!),
    enabled: chosen !== null && night !== undefined,
  });
  // The stored result for the selected model; null when this model has not analysed the night.

  const run = useMutation({
    mutationFn: (model: string) => api.analyse(patient, file, model),
    onSuccess: (result, model) => {
      queryClient.setQueryData(["analysis", patient, file, model], result);
      queryClient.invalidateQueries({ queryKey: ["patient", patient] });
    },
  });

  const { reset } = run;
  useEffect(() => reset(), [chosen, reset]);
  // A failed run's message is cleared when another model is selected.

  useEffect(() => {
    if (night) document.title = `Night ${night.number} · ${patient} · Apnea review`;
  }, [night, patient]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if ((event.target as Element).closest("input, select, textarea, .chart")) return;
      const target = event.key === "ArrowLeft" ? prev : event.key === "ArrowRight" ? next : undefined;
      if (!target) return;
      event.preventDefault();
      navigate(nightPath(patient, target.file));
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [prev, next, patient, navigate]);
  // Left and right arrows open the previous and next night, except while typing, choosing a model
  // or reading the chart, which uses the arrows itself.

  const crumbs = [
    { label: "Patients", to: "/" },
    { label: patient, to: patientPath(patient) },
    { label: night ? `Night ${night.number}` : file },
  ];

  if (nightsQuery.isError) {
    return (
      <>
        <Crumbs items={crumbs.slice(0, 1)} />
        <Placeholder title={`Could not open ${patient}`}>{nightsQuery.error.message}</Placeholder>
      </>
    );
  }
  if (nights && !night) {
    return (
      <>
        <Crumbs items={crumbs} />
        <Placeholder title={`No night ${file} for ${patient}`} />
      </>
    );
  }

  return (
    <>
      <Crumbs items={crumbs} />
      <div className="page-head">
        <div>
          <h1 tabIndex={-1}>{night ? `Night ${night.number}` : "Loading…"}</h1>
          {night && <NightMeta night={night} />}
        </div>
        <nav className="pager" aria-label="Nights">
          <PagerLink patient={patient} night={prev}>{prev ? `‹ Night ${prev.number}` : "‹ Previous"}</PagerLink>
          <PagerLink patient={patient} night={next}>{next ? `Night ${next.number} ›` : "Next ›"}</PagerLink>
        </nav>
      </div>
      <ModelBar action={{
        label: "Analyse",
        busyText: run.isPending ? "Analysing…" : null,
        disabled: !night,
        run: () => chosen && run.mutate(chosen),
      }} />
      {run.isError ? (
        <p className="note error">Analysis failed: {run.error.message}</p>
      ) : kept.data ? (
        <ResultView r={kept.data} patient={patient} file={file} />
      ) : kept.isError ? (
        <p className="note error">{kept.error.message}</p>
      ) : kept.data === null ? (
        <p className="note">Not analysed with this model yet.</p>
      ) : null}
    </>
  );
}
