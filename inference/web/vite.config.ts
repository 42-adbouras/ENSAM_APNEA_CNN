import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
  // `npm run dev` serves the app on port 5173 and forwards /api to the container on port 8000.
});
