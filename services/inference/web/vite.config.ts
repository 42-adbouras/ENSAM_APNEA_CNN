import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: "https://localhost", secure: false } },
  },
  // `npm run dev` serves the app on port 5173 and forwards /api to the nginx container on port 443;
  // secure: false accepts its self-signed certificate.
});
