import { useEffect, useRef } from "react";
import { Link, Route, Routes, useLocation, useMatch } from "react-router";
import { Placeholder } from "./components/common";
import { PatientSwitcher } from "./components/PatientSwitcher";
import { Home } from "./pages/Home";
import { NightPage } from "./pages/NightPage";
import { PatientPage } from "./pages/PatientPage";

export function App() {
  const { pathname } = useLocation();
  const onHome = useMatch("/") !== null;

  const firstRender = useRef(true);
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    window.scrollTo(0, 0);
    document.querySelector<HTMLElement>("#detail h1")?.focus({ preventScroll: true });
  }, [pathname]);
  // After each navigation: back to the top, and keyboard focus on the new page's heading.

  return (
    <>
      <header className="top">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M2 12h4l2-5 4 10 2-5h8" fill="none" stroke="currentColor" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <Link className="brand" to="/">Apnea review</Link>
        <span className="tagline">Overnight ECG recordings</span>
        {!onHome && <PatientSwitcher />}
      </header>
      <main className="layout">
        <section className="panel" id="detail" aria-live="polite">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/patient/:patient" element={<PatientPage />} />
            <Route path="/patient/:patient/night/:night" element={<NightPage />} />
            <Route path="*" element={<Placeholder title="Page not found"><Link to="/">Back to patients</Link></Placeholder>} />
          </Routes>
        </section>
      </main>
    </>
  );
}
// The home page is the patient list, with its own search; every other page has the header search.
