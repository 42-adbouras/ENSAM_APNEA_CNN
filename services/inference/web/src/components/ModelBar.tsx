import type { ReactNode } from "react";
import { useModel } from "../model";
import { Spinner } from "./common";

interface Props {
  action?: {
    label: string;
    busyText: string | null;   // shown in the button while work runs; null when idle
    disabled?: boolean;
    run: () => void;
  };
  children?: ReactNode;        // placed before the model picker, e.g. a search box
}

export function ModelBar({ action, children }: Props) {
  const { models, loading, error, chosen, choose } = useModel();
  const busy = action !== undefined && action.busyText !== null;
  const selectable = !error && models.length > 0;
  const model = models.find(m => m.name === chosen);

  return (
    <div className="model-bar">
      {children}
      <label htmlFor="model" className={children ? "push" : undefined}>Model</label>
      <select id="model" className="select" value={chosen ?? ""} disabled={!selectable || busy}
              onChange={event => choose(event.target.value)}>
        {selectable
          ? models.map(m => (
              <option key={m.name} value={m.name} disabled={m.threshold === null}>
                {m.threshold === null ? `${m.name} (no threshold)` : m.name}
              </option>
            ))
          : <option value="">{loading ? "Loading…" : error ? "Models unavailable" : "No models"}</option>}
      </select>
      <span className="muted">{model?.threshold != null ? `Threshold ${model.threshold.toFixed(2)}` : ""}</span>
      {action && (
        <button type="button" className={busy ? "btn btn-primary busy" : "btn btn-primary"}
                disabled={!model || action.disabled || busy} onClick={action.run}>
          {busy ? <><Spinner />{action.busyText}</> : action.label}
        </button>
      )}
    </div>
  );
}
// A model without a threshold is listed but cannot be selected.
