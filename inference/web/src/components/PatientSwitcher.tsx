import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { patientPath, plural } from "../format";
import { UserIcon } from "./common";

const SHOWN = 8;   // matches listed at once

export function PatientSwitcher() {
  const query = useQuery({ queryKey: ["patients"], queryFn: api.patients });
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);

  const term = text.trim().toLowerCase();
  const matches = (query.data ?? []).filter(p => p.id.toLowerCase().includes(term)).slice(0, SHOWN);

  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      if ((event.target as Element).closest("input, select, textarea")) return;
      event.preventDefault();
      input.current?.focus();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);
  // "/" anywhere outside a form field puts the cursor in this box.

  const go = (id: string) => {
    setText("");
    setOpen(false);
    input.current?.blur();
    navigate(patientPath(id));
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setOpen(true);
      setActive(a => Math.min(a + 1, matches.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActive(a => Math.max(a - 1, 0));
    } else if (event.key === "Enter" && open && matches[active]) {
      event.preventDefault();
      go(matches[active].id);
    } else if (event.key === "Escape") {
      event.preventDefault();
      setText("");
      setOpen(false);
      input.current?.blur();
    }
    // preventDefault stops the browser's own Escape handling of a search box, which clears the
    // text through an input event and would reopen the list.
  };

  return (
    <div className="switcher">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="m20 20-3.5-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
      <input ref={input} type="search" placeholder="Go to patient" autoComplete="off" value={text}
             role="combobox" aria-label="Go to patient" aria-autocomplete="list" aria-controls="switcher-list"
             aria-expanded={open}
             aria-activedescendant={open && matches[active] ? `switcher-option-${active}` : undefined}
             onChange={event => { setText(event.target.value); setOpen(true); setActive(0); }}
             onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onKeyDown={onKeyDown} />
      <kbd aria-hidden="true">/</kbd>
      {open && (
        <ul id="switcher-list" role="listbox" className="switcher-list">
          {matches.length ? matches.map((p, i) => (
            <li key={p.id} id={`switcher-option-${i}`} role="option" aria-selected={i === active}
                onMouseEnter={() => setActive(i)}
                onMouseDown={event => { event.preventDefault(); go(p.id); }}>
              <span className="id"><UserIcon />{p.id}</span>
              <span className="muted">{plural(p.nights, "night")}</span>
            </li>
          )) : <li className="empty">{query.isPending ? "Loading…" : "No patient matches"}</li>}
        </ul>
      )}
    </div>
  );
}
// A combobox: typing filters the patients, arrows move through the matches, Enter opens the
// highlighted one, Escape closes the list. onMouseDown with preventDefault keeps the input from
// losing focus (and closing the list) before the click lands.
