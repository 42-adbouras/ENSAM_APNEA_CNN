import { Fragment, type ReactNode } from "react";
import { Link } from "react-router";
import type { Night } from "../api";
import { formatDuration, formatSize } from "../format";

export const Spinner = () => <span className="spinner" aria-hidden="true" />;

export function Placeholder({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="placeholder">
      <p className="title">{title}</p>
      {children && <p>{children}</p>}
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}{sub && <>{" "}<span className="sub">{sub}</span></>}</dd>
    </div>
  );
}

export interface Crumb {
  label: string;
  to?: string;
}

export function Crumbs({ items }: { items: Crumb[] }) {
  return (
    <nav className="crumbs" aria-label="Breadcrumb">
      {items.map((item, i) => (
        <Fragment key={i}>
          {i > 0 && <span aria-hidden="true">/</span>}
          {item.to ? <Link to={item.to}>{item.label}</Link> : <span aria-current="page">{item.label}</span>}
        </Fragment>
      ))}
    </nav>
  );
}

export function NightMeta({ night }: { night: Night }) {
  return (
    <p className="row-meta">
      <code className="file">{night.file}</code>
      {` · ${formatDuration(night.duration_s)} · ${formatSize(night.size_bytes)}`}
    </p>
  );
}
